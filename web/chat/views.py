from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from chat.utils import call_gpt_stream_with_images  
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_http_methods
from django.http import JsonResponse, FileResponse, HttpResponseNotAllowed, Http404, HttpResponseNotFound, HttpResponseServerError, HttpResponse
from user.models import User
from .models import Chat, Message, Content, MessageImage, UserReview
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dogs.models import DogProfile, DogBreed
from django.contrib.auth.decorators import login_required
from user.utils import get_logged_in_user
from collections import defaultdict
from django.http import HttpResponseForbidden
from datetime import date, timedelta, time
from asgiref.sync import sync_to_async
import uuid
import requests
import json
import pandas as pd
import base64
from django.template.loader import render_to_string, get_template
import os
from django.conf import settings
from .report_utils.gpt_report import build_prompt, generate_response, clean_and_split
from .report_utils.report_pdf import generate_pdf_from_context
from rest_framework.decorators import api_view
from rest_framework.response import Response
from chat.utils import get_image_response
from datetime import datetime
from django.utils.timezone import make_aware
from urllib.parse import quote
from django.urls import reverse
import mimetypes
from urllib.parse import unquote
from dogs.models import PersonalityResult
from chat.utils import format_dog_info
import pytz
from pathlib import Path
from decouple import Config, RepositoryEnv

def chat_entry(request):
    if request.session.get('guest'):
        return redirect('chat:main')

    elif request.session.get('user_id'):
        dog_id = request.session.get('current_dog_id')
        if dog_id:
            return redirect('chat:chat_member', dog_id=dog_id)
        else:
            return redirect('dogs:dog_info_join')

    else:
        return redirect('user:home')
    
def is_guest_user(request):
    return request.session.get("guest", False)

def get_user_id(request):
    return request.session.get("guest_user_id") if is_guest_user(request) else request.session.get("user_id")

def is_chat_owner(request, chat):
    current_user_id = get_user_id(request)
    return str(chat.user_id) == str(current_user_id)

    
def group_chats_by_date(chat_list):
    today = date.today()
    yesterday = today - timedelta(days=1)
    grouped = defaultdict(list)

    for chat in chat_list:
        created = chat.created_at.date()
        if created == today:
            label = "오늘"
        elif created == yesterday:
            label = "어제"
        else:
            label = created.strftime("%Y.%m.%d")
        grouped[label].append(chat)

    return dict(grouped)

def chat_member_view(request, dog_id):
    user = get_logged_in_user(request)
    if not user:
        return redirect('user:home')

    dog = get_object_or_404(DogProfile, id=dog_id, user=user)

    dog_list = DogProfile.objects.filter(user=user).order_by('created_at')

    chat_list = Chat.objects.filter(dog=dog).order_by('-created_at')
    grouped_chat_list = group_chats_by_date(chat_list)
    current_chat = Chat.objects.filter(dog=dog).order_by('-created_at').first()
    messages = Message.objects.filter(chat=current_chat).order_by('created_at') if current_chat else []

    request.session['current_dog_id'] = dog.id

    return render(request, 'chat/chat.html', {
        'grouped_chat_list': grouped_chat_list,
        'chat_list': chat_list,
        'current_chat': current_chat,
        'chat_messages': messages,
        'is_guest': False,
        'user_email': user.email,
        'dog': dog,
        'dog_id': dog.id,    
        'dog_list': dog_list,
        'can_generate_report': False,
        'hide_report_button': True,
    })


@csrf_exempt
@require_http_methods(["GET", "POST"])
def guest_profile_register(request):
    if request.method == 'GET':
        request.session.flush()
        request.session['guest'] = True

        guest_email = f"guest_{uuid.uuid4().hex[:10]}@example.com"
        guest_user = User.objects.create(email=guest_email, password='guest_pw')
        request.session['guest_user_id'] = str(guest_user.id)
        request.session['user_email'] = guest_email

        return redirect('chat:main')

    elif request.method == 'POST':
        guest_name = request.POST.get("guest_name", "").strip()
        guest_breed = request.POST.get("guest_breed", "").strip()

        if not guest_name or not guest_breed:
            return redirect('chat:main')

        request.session["guest_dog_name"] = guest_name
        request.session["guest_dog_breed"] = guest_breed

        guest_user_id = request.session.get("guest_user_id")
        user = User.objects.get(id=guest_user_id)

        chat = Chat.objects.create(user=user, dog=None, chat_title="비회원 상담 시작")
        welcome_message = f"{guest_name}의 상담을 시작해볼까요? 😊"
        Message.objects.create(chat=chat, sender="bot", message=welcome_message)
        request.session["current_chat_id"] = str(chat.id)

        return redirect('chat:chat_talk_detail', chat_id=chat.id)

    return HttpResponseNotAllowed(['GET', 'POST'])


def stream_answer(answer):
    def generate():
        for char in answer:
            yield char
    return StreamingHttpResponse(generate(), content_type='text/plain; charset=utf-8')


@require_http_methods(["GET", "POST"])
def chat_member_talk_detail(request, dog_id, chat_id):
    user_id = request.session.get("user_id")
    if not user_id:
        return JsonResponse({'error': '로그인 필요'}, status=401)  

    try:
        user = User.objects.get(id=uuid.UUID(user_id))
    except (User.DoesNotExist, ValueError):
        return JsonResponse({'error': '사용자를 찾을 수 없습니다.'}, status=404) 

    dog = DogProfile.objects.filter(id=dog_id).first()
    if not dog or str(dog.user.id) != str(user.id):
        return HttpResponseForbidden("접근 권한이 없습니다.")

    chat = Chat.objects.filter(id=chat_id, dog=dog).first()
    if not chat:
        return HttpResponseForbidden("접근 권한이 없습니다.")

    if request.method == "POST":
        message = request.POST.get("message", "").strip()
        image_files = request.FILES.getlist("images")

        if not message and not image_files:
            return JsonResponse({'error': '메시지나 이미지를 입력해주세요.'}, status=400)

        if message:
            user_message = Message.objects.create(chat=chat, sender='user', message=message)
        elif image_files:
            user_message = Message.objects.create(chat=chat, sender='user', message="[이미지 전송]")

        for img in image_files[:3]:
            try:
                MessageImage.objects.create(message=user_message, image=img)
            except Exception:
                pass

        if image_files:
            async def gpt_stream():
                final_answer = ""
                try:
                    dog_info = await sync_to_async(get_dog_info)(dog, chat=chat, user_id=user.id)
                    profile_text = await sync_to_async(format_dog_info)(dog_info)
                    full_question = f"[보호자 질문]\n{message}\n\n[반려견 프로필]\n{profile_text}"

                    async for chunk in call_gpt_stream_with_images(image_files, full_question):
                        final_answer += chunk
                        yield chunk
                except Exception as e:
                    yield f"\n[에러 발생: {str(e)}]"
                finally:
                    await sync_to_async(Message.objects.create)(
                        chat=chat,
                        sender='bot',
                        message=final_answer.strip()
                    )
            return StreamingHttpResponse(gpt_stream(), content_type="text/plain")

        elif message:
            user_info = get_dog_info(dog)
            answer = call_runpod_api(message, user_info)
            Message.objects.create(chat=chat, sender='bot', message=answer)

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return stream_answer(answer)
            else:
                return JsonResponse({'response': answer, 'chat_id': chat.id}, json_dumps_params={'ensure_ascii': False})

    messages = Message.objects.filter(chat=chat).prefetch_related("images").order_by('created_at')
    chat_list = Chat.objects.filter(dog=dog).order_by('-created_at')
    grouped_chat_list = group_chats_by_date(chat_list)

    dog_list = DogProfile.objects.filter(user=user).order_by('created_at')

    return render(request, "chat/chat_talk.html", {
        "messages": messages,
        "current_chat": chat,
        "chat_list": chat_list,
        "grouped_chat_list": grouped_chat_list,
        "user_email": user.email,
        "is_guest": False,
        "now_time": timezone.localtime().strftime("%I:%M %p").lower(),
        "dog": dog,
        "dog_id": dog.id,
        "dog_list": dog_list,
        'can_generate_report': True,
    })


def chat_main(request):
    dog_id = None
    is_guest = request.session.get("guest", False)
    user_id = request.session.get("user_id")
    guest_user_id = request.session.get("guest_user_id")
    user_email = request.session.get("user_email")
    current_dog_id = request.session.get("current_dog_id")
    current_chat_id = request.session.get("current_chat_id")

    guest_name = request.session.get("guest_dog_name")
    guest_breed = request.session.get("guest_dog_breed")

    dog_breeds = DogBreed.objects.all().order_by("name")

    if is_guest and (not guest_name or not guest_breed):
        return render(request, "chat/chat.html", {
            "show_guest_info_form": True,
            "is_guest": True,
            "dog_breeds": dog_breeds,
            'hide_report_button': True,
        })

    chat_list, current_chat, messages = [], None, []

    if user_id and not is_guest:
        try:
            user = User.objects.get(id=user_id)
            chat_list = Chat.objects.filter(dog__user=user).order_by('-created_at')

            if current_dog_id:
                current_chat = Chat.objects.filter(dog__id=current_dog_id).first()
                dog_id = current_dog_id
            else:
                current_chat = chat_list.first()
                if current_chat and current_chat.dog:
                    dog_id = current_chat.dog.id
                    request.session["current_dog_id"] = current_chat.dog.id

        except User.DoesNotExist:
            return redirect('user:home')

    elif is_guest and guest_user_id:
        try:
            user = User.objects.get(id=guest_user_id)
            chat_list = Chat.objects.filter(dog=None, user=user).order_by('-created_at')

            if current_chat_id:
                current_chat = Chat.objects.filter(id=current_chat_id, user=user).first()

            if not current_chat:
                current_chat = chat_list.first()

            if not current_chat:
                current_chat = Chat.objects.create(user=user, dog=None, chat_title="비회원 상담 시작")
                Message.objects.create(chat=current_chat, sender="bot", message=f"{guest_name}의 상담을 시작해볼까요? 😊")
                chat_list = Chat.objects.filter(dog=None, user=user).order_by('-created_at')

            request.session["current_chat_id"] = str(current_chat.id)

        except User.DoesNotExist:
            return redirect('user:home')

    else:
        return redirect('user:home')

    if current_chat:
        messages = Message.objects.filter(chat=current_chat).order_by('created_at')

    return render(request, 'chat/chat.html', {
        'chat_list': chat_list,
        'current_chat': current_chat,
        'chat_messages': messages,
        'is_guest': is_guest,
        'user_email': user_email,
        'guest_dog_name': guest_name,
        'guest_dog_breed': guest_breed,
        'dog_breeds': dog_breeds,
        'dog_id': dog_id,
        'show_guest_info_form': False,
        'show_login_notice': is_guest,
        'hide_report_button': True
    })

def chat_switch_dog(request, dog_id):
    user_id = request.session.get("user_id")
    if not user_id:
        return redirect("user:home")

    dog = get_object_or_404(DogProfile, id=dog_id, user_id=user_id)

    return redirect('chat:chat_member', dog_id=dog.id)
    
def get_dog_info(dog, chat=None, user_id=None):
    if chat is not None:
        chat_history, prev_q, prev_a = get_chat_history(chat)
    else:
        chat_history, prev_q, prev_a = [], None, None

    def safe(v, default="모름"):
        if v is None:
            return default
        if isinstance(default, str) and isinstance(v, int):
            return str(v)
        return v
    
    try:
        personality = PersonalityResult.objects.get(dog=dog)
        personality_character = personality.character
    except PersonalityResult.DoesNotExist:
        personality_character = ""


    info = {
        "name": safe(dog.name, ""),
        "breed": safe(dog.breed.name if dog.breed else None),
        "age": safe(dog.age),
        "gender": safe(dog.gender),
        "neutered": safe(dog.neutered),
        "disease": safe("있음" if dog.disease_history else "없음"),
        "disease_desc": safe(dog.disease_history, ""),
        "period": safe(dog.living_period),
        "housing": safe(dog.housing_type),
        "chat_history": chat_history,
        "prev_q": prev_q,
        "prev_a": prev_a,
        "prev_cate": None,
        "is_first_question": len(chat_history) == 0,
        "user_id": user_id if user_id else (str(dog.user.id) if hasattr(dog, "user") else "unknown"),
        "personality": personality_character
    }
    return info

def get_minimal_guest_info(session, chat=None, user_id=None):
    name = session.get("guest_dog_name", "비회원견")
    breed = session.get("guest_dog_breed", "견종 정보 없음")
    if chat is not None:
        chat_history, prev_q, prev_a = get_chat_history(chat)
    else:
        chat_history, prev_q, prev_a = [], None, None

    info = {
        "name": name,
        "breed": breed,
        "age": "모름",
        "gender": "모름",
        "neutered": "모름",
        "disease": "모름",
        "disease_desc": "",
        "period": "모름",
        "housing": "모름",
        "chat_history": chat_history,
        "prev_q": prev_q,
        "prev_a": prev_a,
        "prev_cate": None,
        "is_first_question": len(chat_history) == 0,
        "user_id": user_id if user_id else session.get("guest_user_id", "guest")
    }
    return info

def get_chat_history(chat):
    past_msgs = Message.objects.filter(chat=chat).order_by("created_at")
    chat_history = [
        {"role": "user" if m.sender == "user" else "assistant", "content": m.message}
        for m in past_msgs
    ]
    prev_q, prev_a = None, None
    for i in range(len(chat_history) - 2, -1, -2):
        if chat_history[i]["role"] == "user" and chat_history[i + 1]["role"] == "assistant":
            prev_q = chat_history[i]["content"]
            prev_a = chat_history[i + 1]["content"]
            break
    return chat_history, prev_q, prev_a

BASE_DIR = Path(__file__).resolve().parent.parent
env_path = os.path.join(BASE_DIR, '.env')
config = Config(RepositoryEnv(env_path))

def call_runpod_api(message, dog_info):
    try:
        api_host = config("RUNPOD_API_HOST")
        api_url = f"http://{api_host}/chat"
        payload = {
            "message": message,
            "dog_info": dog_info
        }
        res = requests.post(api_url, json=payload, timeout=120)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "⚠️ 응답이 없습니다.")
    except Exception as e:
        return f"일시적인 오류로 답변을 제공하지 못하니 잠시 후 다시 시도해 주세요 🐶"



@require_POST
@csrf_exempt
def chat_send(request):
    is_guest = is_guest_user(request)
    user_id = get_user_id(request)

    if not user_id:
        return redirect('user:login')

    user = get_object_or_404(User, id=user_id)

    message = request.POST.get("message", "").strip()
    image_files = request.FILES.getlist("images")

    if not message and not image_files:
        return redirect(request.META.get('HTTP_REFERER', '/'))

    if is_guest:
        chat_id = request.session.get("current_chat_id")
        chat = Chat.objects.filter(id=chat_id).first()

        if chat and not is_chat_owner(request, chat):
            return JsonResponse({"error": "비회원 권한 없음"}, status=403)

        if not chat:
            chat = Chat.objects.create(user=user, dog=None, chat_title=message[:20] if message else "비회원 상담")
            request.session["current_chat_id"] = str(chat.id)

        user_message = Message.objects.create(
            chat=chat,
            sender="user",
            message=message if message else "[이미지 전송]"
        )

        for img in image_files[:3]:
            try:
                MessageImage.objects.create(message=user_message, image=img)
            except Exception:
                pass

        return redirect('chat:chat_talk_detail', chat_id=chat.id)

    current_dog_id = request.session.get("current_dog_id")
    dog = DogProfile.objects.filter(id=current_dog_id, user=user).first()

    if not dog:
        return redirect('dogs:dog_info_join')

    chat = Chat.objects.create(
        dog=dog,
        user=user,
        chat_title=message[:20] if message else "상담 시작"
    )

    url = reverse('chat:chat_member_talk_detail', args=[dog.id, chat.id])
    return redirect(f"{url}?just_sent=1&last_msg={quote(message)}")


@require_POST
@csrf_exempt
def chat_member_delete(request, dog_id, chat_id):
    try:
        if request.method == "POST":
            chat = get_object_or_404(Chat, id=chat_id, dog_id=dog_id)

            if not is_chat_owner(request, chat):
                return JsonResponse({'status': 'unauthorized'}, status=403)

            chat.delete()
            return JsonResponse({'status': 'ok'})

    except Chat.DoesNotExist:
        return JsonResponse({'error': 'Invalid method'}, status=405)
    

@require_POST
@csrf_exempt
@require_POST
@csrf_exempt
def chat_member_update_title(request, dog_id, chat_id):
    try:
        chat = get_object_or_404(Chat, id=chat_id, dog_id=dog_id)

        if not is_chat_owner(request, chat):
            return JsonResponse({'status': 'unauthorized'}, status=403)

        data = json.loads(request.body)
        new_title = data.get('title', '').strip()

        if new_title:
            chat.chat_title = new_title
            chat.save()
            return JsonResponse({'status': 'ok'})
        else:
            return JsonResponse({'status': 'empty_title'}, status=400)

    except json.JSONDecodeError:
        return JsonResponse({'status': 'invalid_json'}, status=400)

    except Chat.DoesNotExist:
        return JsonResponse({'status': 'not_found'}, status=404)
    

@require_http_methods(["GET", "POST"])
def chat_talk_view(request, chat_id):
    is_guest = is_guest_user(request)
    user_id = get_user_id(request)
    if not user_id:
        return redirect('user:home')

    chat = get_object_or_404(Chat, id=chat_id)

    if not is_chat_owner(request, chat):
        return HttpResponseForbidden("접근 권한이 없습니다.")

    user_email = request.session.get("user_email")
    current_dog_id = request.session.get("current_dog_id")
    user_id = request.session.get("guest_user_id") if is_guest else request.session.get("user_id")

    try:
        chat = Chat.objects.get(id=chat_id)
    except Chat.DoesNotExist:
        return JsonResponse({'error': '채팅을 찾을 수 없습니다.'}, status=404) 

    if not is_guest:
        if not user_id or not chat.user or str(chat.user.id) != str(user_id):
            return JsonResponse({'error': '권한이 없습니다.'}, status=403)  

    if request.method == "POST":
        message_text = request.POST.get("message", "").strip()
        image_files = request.FILES.getlist("images")

        if not message_text and not image_files:
            return JsonResponse({'error': '메시지나 이미지가 필요합니다.'}, status=400)  

        user_message = Message.objects.create(
            chat=chat,
            sender='user',
            message=message_text if message_text else "[이미지 전송]"
        )

        for img in image_files[:3]:
            try:
                MessageImage.objects.create(message=user_message, image=img)
            except Exception:
                pass

        if image_files:
            user_info = get_dog_info(dog, chat=chat, user_id=user_id)
            answer = get_image_response(image_files, user_message, user_info)
        else:
            if is_guest:
                user_info = get_minimal_guest_info(request.session, chat=chat, user_id=user_id)
            else:
                chat_history, prev_q, prev_a = get_chat_history(chat)
                user_info = get_dog_info(chat.dog, chat=chat, user_id=user_id)
                user_info.update({
                    "chat_history": chat_history,
                    "prev_q": prev_q,
                    "prev_a": prev_a,
                    "prev_cate": None,
                    "is_first_question": len(chat_history) == 0,
                })

            answer = call_runpod_api(message_text, user_info)

        Message.objects.create(chat=chat, sender='bot', message=answer)

        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return stream_answer(answer)
        else:
            return JsonResponse({'response': answer, 'chat_id': chat.id}, json_dumps_params={'ensure_ascii': False})

    messages = Message.objects.filter(chat=chat).prefetch_related("images").order_by('created_at')
    chat_list = Chat.objects.filter(user__id=user_id).order_by('-created_at') if not is_guest else []
    now_time = timezone.localtime().strftime("%I:%M %p").lower()

    dog = chat.dog if not is_guest else None
    dog_list = DogProfile.objects.filter(user__id=user_id).order_by('created_at') if dog else []

    return render(request, "chat/chat_talk.html", {
        "messages": messages,
        "current_chat": chat,
        "chat_list": chat_list,
        "user_email": user_email,
        "is_guest": is_guest,
        "now_time": now_time,
        "dog": dog,
        "dog_list": dog_list
    })



@require_http_methods(["GET"])
def recommend_content(request, chat_id):
    if not request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({"error": "Invalid request"}, status=400)

    chat = get_object_or_404(Chat, id=chat_id)

    if not is_chat_owner(request, chat):
        return JsonResponse({
            "error": "접근 권한이 없습니다.",
            "cards_html": "",
            "has_recommendation": False
        }, status=403)

    history = Message.objects.filter(chat=chat).order_by("created_at")
    chat_history = [
        {"role": "user" if m.sender == "user" else "assistant", "content": m.message}
        for m in history
    ]

    contents = Content.objects.all().values("title", "body", "reference_url", "image_url")
    df = pd.DataFrame.from_records(contents)

    if df.empty:
        return JsonResponse({
            "cards_html": "",
            "has_recommendation": False
        })

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["body"])
    chat_text = "\n".join([m["content"] for m in chat_history if m["role"] in ["user", "assistant"]])

    if not chat_text.strip():
        return JsonResponse({
            "cards_html": "",
            "has_recommendation": False
        })

    user_vector = vectorizer.transform([chat_text])
    cosine_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = cosine_scores.argsort()[-3:][::-1]
    top_contents = df.iloc[top_indices]

    html = '''
    <div class="recommend-content">
    <p style="font-weight:400; margin: 0 0 12px 0; font-size:15px;">
    🐾 반려견의 마음을 이해하는 데 도움 되는 이야기들이에요:
    </p>
    <div style="display:flex; flex-direction:column; gap:12px;">
    '''

    for item in top_contents.to_dict(orient="records"):
        image_url = item['image_url']
        has_image = image_url and image_url.strip().startswith("http")

        if has_image:
            html += f'''
            <a href="{item['reference_url']}" target="_blank" class="recommend-card-link">
            <div class="recommend-card with-image">
                <div class="card-content-section">
                <p class="recommend-title">{item['title']}</p>
                <p class="recommend-description">{item['body'][:80]}···</p>
                <span class="recommend-link-text">👉 자세히 보기</span>
                </div>
            </div>
            </a>
            '''
        else:
            html += f'''
            <a href="{item['reference_url']}" target="_blank" class="recommend-card-link">
            <div class="recommend-card with-image">
                <div class="card-content-section">
                <p class="recommend-title">{item['title']}</p>
                <p class="recommend-description">{item['body'][:80]}···</p>
                <span class="recommend-link-text">👉 자세히 보기</span>
                </div>
            </div>
            </a>
            '''

    html += '</div></div>'

    Message.objects.create(
        chat=chat,
        sender="bot",
        message=html,
        created_at=timezone.now()
    )

    return JsonResponse({
        "cards_html": html,
        "has_recommendation": True
    })

@csrf_exempt
@require_POST
def submit_review(request):
    try:
        data = json.loads(request.body)
        chat_id = data.get('chat_id')
        score = data.get('review_score')
        review = data.get('review')

        if not chat_id or score is None:
            return JsonResponse({'status': 'invalid_input'}, status=400)

        chat = get_object_or_404(Chat, id=chat_id)

        if not is_chat_owner(request, chat):
            return JsonResponse({'status': 'unauthorized'}, status=403)

        UserReview.objects.create(
            chat=chat,
            review_score=score,
            review=review
        )
        return JsonResponse({'status': 'ok'})

    except (json.JSONDecodeError, KeyError):
        return JsonResponse({'status': 'invalid_json'}, status=400)
    


def load_chat_and_profile(chat_id, start_date, end_date):
    try:
        chat = Chat.objects.select_related("dog").get(id=chat_id)
    except Chat.DoesNotExist:
        return None, None

    dog = chat.dog
    if not dog:
        return None, None

    dog_dict = {
        "name": dog.name,
        "age": dog.age,
        "breed_name": dog.breed.name if dog.breed else "알 수 없음",
        "gender": dog.gender,
        "neutered": dog.neutered,
        "disease_history": dog.disease_history,
        "living_period": dog.living_period,
        "housing_type": dog.housing_type,
        "image": dog.profile_image.url if dog.profile_image else None,
    }

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        start_dt = make_aware(datetime.combine(start_dt, time.min)) 
        end_dt = make_aware(datetime.combine(end_dt, time.max))   
    except ValueError:
        return dog_dict, []

    messages = Message.objects.filter(
        chat_id=chat_id,
        created_at__range=(start_dt, end_dt)
    ).order_by("created_at")

    history = [
        {"role": "user" if msg.sender == "user" else "assistant", "content": msg.message}
        for msg in messages if msg.message
    ]
    return dog_dict, history


def chat_report_feedback_view(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id)
    return render(request, 'chat/chat_report_feedback.html', {
        "chat_id": chat_id,
    })

def get_base64_image(image_path_or_url):
    try:
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url, timeout=5)
            if response.status_code == 200:
                mime_type = response.headers.get("Content-Type", "image/jpeg")
                base64_str = base64.b64encode(response.content).decode("utf-8")
                return base64_str, mime_type
            else:
                print(f"[오류] S3 이미지 요청 실패: {response.status_code}")
                return None, None
        else:
            if image_path_or_url.startswith("media/"):
                image_path_or_url = image_path_or_url[len("media/"):]
            elif image_path_or_url.startswith("/media/"):
                image_path_or_url = image_path_or_url[len("/media/"):]

            full_path = os.path.join(settings.MEDIA_ROOT, image_path_or_url)

            with open(full_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")
                mime_type, _ = mimetypes.guess_type(full_path)
                return encoded, mime_type or "image/jpeg"
    except Exception as e:
        print(f"[오류] 이미지 인코딩 실패: {str(e)}")
        return None, None

@api_view(['POST'])
def generate_report(request):
    data = request.data
    chat_id = data.get("chat_id")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    print("📩 받은 데이터:", data)

    if not (chat_id and start_date and end_date):
        print("❌ 필수 파라미터 누락")
        return Response({"error": "필수 값 누락"}, status=400)

    dog, history = load_chat_and_profile(chat_id, start_date, end_date)
    if not dog or not history:
        print(f"❌ dog 또는 history 없음 - chat_id: {chat_id}")
        return Response({"error": "해당 chat_id에 대한 데이터가 없습니다."}, status=404)

    try:
        messages = build_prompt(dog, history)
        raw_output = generate_response(messages)
        intro, advice, next_, is_split_success = clean_and_split(raw_output)
        print("✅ GPT 응답 정상 처리 완료")
    except Exception as e:
        print("❌ GPT 처리 중 오류:", str(e))
        return Response({"error": f"GPT 처리 중 오류: {str(e)}"}, status=500)

    base64_img = None
    mime_type = None
    if dog.get("image"):
        try:
            image_path = dog["image"]
            cleaned_image_path = unquote(image_path.replace("/media/", ""))
            base64_img, mime_type = get_base64_image(cleaned_image_path)

            if not base64_img:
                print("⚠️ 이미지 인코딩 실패 또는 이미지 없음:", cleaned_image_path)
        except Exception as e:
            print("❌ 이미지 인코딩 중 오류:", str(e))
            base64_img = None
            mime_type = None

    context = {
        "dog_name": dog["name"],
        "age": dog["age"],
        "gender_display": dog["gender"],
        "neutered": dog["neutered"],
        "disease_history": dog["disease_history"],
        "living_period": dog["living_period"],
        "housing_type": dog["housing_type"],
        "image": base64_img,
        "image_mime_type": mime_type,
        "start_date": start_date,
        "end_date": end_date,
        "intro_text": intro,
        "advice_text": advice,
        "next_text": next_,
        "is_split_success": is_split_success,
        "full_text": raw_output,
        "request": request,
    }

    try:
        from django.template.loader import render_to_string
        html_str = render_to_string("chat/report_template.html", context)
        debug_path = "/tmp/last_report.html"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"📄 HTML 디버깅 파일 저장됨: {debug_path}")

        pdf_path = generate_pdf_from_context(context, pdf_filename=f"report_{chat_id}.pdf")
        print(f"✅ PDF 생성 성공: {pdf_path}")

        request.session[f"pdf_path_{chat_id}"] = pdf_path
        return Response({"status": "success"})

    except Exception as e:
        print("❌ PDF 생성 실패:", str(e))
        return Response({"error": f"PDF 생성 실패: {str(e)}"}, status=500)
    

def download_report_pdf(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id)

    if not is_chat_owner(request, chat):
        raise Http404("다운로드 권한이 없습니다.")

    pdf_path = request.session.get(f"pdf_path_{chat_id}")
    if not pdf_path or not os.path.exists(pdf_path):
        raise Http404("리포트 파일이 존재하지 않거나 세션이 만료되었습니다.")

    try:
        pdf_file = open(pdf_path, "rb")
        response = FileResponse(pdf_file, as_attachment=True, filename=f"report_{chat_id}.pdf")

        def cleanup():
            try:
                pdf_file.close()
                os.remove(pdf_path)
                print("🧹 다운로드 후 PDF 삭제 완료")
            except Exception:
                pass

        response.close = cleanup

        return response

    except Exception as e:
        print("❌ PDF 전송 에러:", str(e))
        raise Http404("리포트 다운로드 중 문제가 발생했습니다.")


@api_view(['GET'])
def check_report_status(request):
    return Response({"status": "done"})