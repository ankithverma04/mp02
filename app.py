from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import re
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import io
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
try:
    import google.generativeai as genai
except Exception:
    genai = None

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
database_url = os.getenv('DATABASE_URL', 'sqlite:///interview_platform.db')
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads', 'videos')

# Configure connection pool for PostgreSQL to handle SSL and connection timeouts
if database_url.startswith('postgresql'):
    # Parse existing SSL mode from URL if present, otherwise default to require
    import urllib.parse
    parsed = urllib.parse.urlparse(database_url)
    query_params = urllib.parse.parse_qs(parsed.query)
    ssl_mode = query_params.get('sslmode', ['require'])[0] if query_params.get('sslmode') else 'require'
    
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,  # Verify connections before using them (reconnects if closed)
        'pool_recycle': 300,     # Recycle connections after 5 minutes
        'pool_size': 5,          # Number of connections to maintain
        'max_overflow': 10,      # Maximum overflow connections
        'connect_args': {
            'connect_timeout': 10,  # Connection timeout in seconds
            'sslmode': ssl_mode      # Use SSL mode from URL or default to require
        }
    }

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
Migrate(app, db)

# Add custom Jinja2 filter for JSON parsing
@app.template_filter('from_json')
def from_json_filter(value):
    """Parse JSON string to Python object"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value

# Ensure Flask's session is always available in templates (even if shadowed)
@app.context_processor
def inject_session():
    """Make Flask's session available in all templates"""
    from flask import session as flask_session
    return {'flask_session': flask_session}


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    interviews = db.relationship('Interview', backref='user', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.email}>'


class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    domain = db.Column(db.String(50), nullable=False)  # Python, Java, SQL, etc.
    score = db.Column(db.Float, nullable=False)  # Score out of 100
    total_questions = db.Column(db.Integer, nullable=False, default=0)
    correct_answers = db.Column(db.Integer, nullable=False, default=0)
    performance_summary = db.Column(db.Text, nullable=True)  # Brief performance feedback
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<Interview {self.id} - {self.domain}>'
    
    @property
    def accuracy_rate(self):
        """Calculate accuracy rate as percentage"""
        if self.total_questions == 0:
            return 0
        return round((self.correct_answers / self.total_questions) * 100, 1)


class InterviewSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    domain = db.Column(db.String(50), nullable=False)
    started_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime, nullable=True)
    current_index = db.Column(db.Integer, nullable=False, default=0)
    total_questions = db.Column(db.Integer, nullable=False, default=5)


class QuestionAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('interview_session.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    correct_answer = db.Column(db.Text, nullable=True)
    user_answer = db.Column(db.Text, nullable=True)
    is_correct = db.Column(db.Boolean, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    voice_clarity_score = db.Column(db.Float, nullable=True)
    voice_confidence_score = db.Column(db.Float, nullable=True)
    voice_tone_analysis = db.Column(db.Text, nullable=True)
    # NLP evaluation scores
    keyword_score = db.Column(db.Float, nullable=True)  # Keyword matching score (0-100)
    semantic_score = db.Column(db.Float, nullable=True)  # Semantic similarity score (0-100)
    grammar_score = db.Column(db.Float, nullable=True)  # Grammar score (0-100)
    final_nlp_score = db.Column(db.Float, nullable=True)  # Weighted final score (0-100)
    nlp_feedback = db.Column(db.Text, nullable=True)  # Generated feedback text

    session_rel = db.relationship('InterviewSession', backref='attempts')


class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    email_notifications = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class InterviewVideo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('interview_session.id'), nullable=True)
    domain = db.Column(db.String(50), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.Float, nullable=True)  # Duration in seconds
    file_size = db.Column(db.Integer, nullable=True)  # File size in bytes
    emotion_analysis = db.Column(db.Text, nullable=True)  # JSON string with emotion data
    engagement_score = db.Column(db.Float, nullable=True)  # Overall engagement score 0-10
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    
    user_rel = db.relationship('User', backref='videos')
    session_rel = db.relationship('InterviewSession', backref='videos')
    
    def __repr__(self):
        return f'<InterviewVideo {self.id} - {self.filename}>'


class EmotionAnalysis(db.Model):
    """Store detailed emotion analysis results from DeepFace"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('interview_video.id'), nullable=True)
    session_id = db.Column(db.Integer, db.ForeignKey('interview_session.id'), nullable=True)
    # Emotion percentages (0-100)
    neutral = db.Column(db.Float, nullable=True, default=0.0)
    happy = db.Column(db.Float, nullable=True, default=0.0)
    sad = db.Column(db.Float, nullable=True, default=0.0)
    angry = db.Column(db.Float, nullable=True, default=0.0)
    fear = db.Column(db.Float, nullable=True, default=0.0)
    disgust = db.Column(db.Float, nullable=True, default=0.0)
    surprise = db.Column(db.Float, nullable=True, default=0.0)
    # Summary metrics
    engagement_score = db.Column(db.Float, nullable=True)  # 0-100
    dominant_emotion = db.Column(db.String(50), nullable=True)
    frames_analyzed = db.Column(db.Integer, nullable=True, default=0)
    faces_detected = db.Column(db.Integer, nullable=True, default=0)
    analyzed_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    user_rel = db.relationship('User', backref='emotion_analyses')
    video_rel = db.relationship('InterviewVideo', backref='emotion_analysis_records')
    session_rel = db.relationship('InterviewSession', backref='emotion_analyses')
    
    def __repr__(self):
        return f'<EmotionAnalysis {self.id} - {self.dominant_emotion}>'


def validate_email(email):
    """Validate email format using regex"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """Validate password: at least 8 chars, one number, one special char"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, ""


def evaluate_answer_with_gemini(model_answer: str, candidate_answer: str) -> dict:
    """
    Evaluate candidate answer using Gemini AI.
    Returns dict with score (0-100) and feedback.
    """
    client = _configure_gemini()
    if not client:
        return {
            'score': 0,
            'feedback': 'Gemini API not available for evaluation.'
        }
    
    prompt = f"""You are an expert technical interviewer evaluating a candidate's answer.

Expected Answer: "{model_answer}"

Candidate's Answer: "{candidate_answer}"

Evaluate the candidate's answer and provide:
1. A score from 0-100 based on:
   - Accuracy and correctness (40%)
   - Completeness and detail (30%)
   - Clarity and communication (20%)
   - Understanding demonstrated (10%)
2. Constructive feedback (1-2 sentences)

Be lenient with scoring:
- Give partial credit for partial understanding
- Reward honest "I don't know" responses with at least 30-40 points
- Don't penalize minimal answers too harshly if they're correct
- Focus on whether the core concept is understood

Return ONLY a JSON object with this exact format:
{{
    "score": <number 0-100>,
    "feedback": "<constructive feedback message>"
}}

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, just the JSON object."""
    
    try:
        import json as _json
        model = genai.GenerativeModel('gemini-1.5-flash')
        generation_config = {
            'temperature': 0.3,  # Low temperature for consistent evaluation
            'max_output_tokens': 200,
        }
        resp = model.generate_content(prompt, generation_config=generation_config)
        text = resp.text.strip()
        
        # Remove markdown code blocks if present
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        # Extract JSON
        if '{' in text and '}' in text:
            json_str = text[text.find('{'):text.rfind('}')+1]
            data = _json.loads(json_str)
            
            score = float(data.get('score', 0))
            feedback = data.get('feedback', 'No feedback provided.')
            
            # Ensure score is in valid range
            score = max(0, min(100, score))
            
            return {
                'score': round(score, 1),
                'feedback': feedback
            }
    except Exception as e:
        # Fallback: return neutral score
        return {
            'score': 50,
            'feedback': 'Unable to evaluate with Gemini. Using alternative method.'
        }
    
    return {
        'score': 50,
        'feedback': 'Unable to evaluate with Gemini. Using alternative method.'
    }


def evaluate_answer_with_nlp(model_answer: str, candidate_answer: str) -> dict:
    """
    Evaluate candidate answer using NLP techniques:
    - Keyword matching (30% weight)
    - Semantic similarity (50% weight)
    - Grammar check (20% weight)
    
    Returns dict with scores and feedback.
    """
    if not model_answer or not candidate_answer:
        return {
            'keyword_score': 0,
            'semantic_score': 0,
            'grammar_score': 0,
            'final_score': 0,
            'feedback': 'Unable to evaluate: missing answer or model answer.'
        }
    
    results = {
        'keyword_score': 0,
        'semantic_score': 0,
        'grammar_score': 0,
        'final_score': 0,
        'feedback': ''
    }
    
    # Check for "I don't know" or similar honest responses
    candidate_lower = candidate_answer.lower().strip()
    dont_know_phrases = [
        "i don't know", "i don't", "i do not know", "i do not", 
        "don't know", "do not know", "not sure", "unsure",
        "i'm not sure", "i am not sure", "no idea", "have no idea",
        "unfamiliar", "not familiar", "haven't learned", "have not learned"
    ]
    
    is_honest_response = any(phrase in candidate_lower for phrase in dont_know_phrases)
    
    # For honest "I don't know" responses, be more lenient
    if is_honest_response:
        # Give partial credit for honesty and clear communication
        results['keyword_score'] = 30  # Some credit for being honest
        results['semantic_score'] = 40  # Partial credit for clear communication
        results['grammar_score'] = 90   # Good grammar for honest responses
        results['final_score'] = round(
            results['keyword_score'] * 0.3 +
            results['semantic_score'] * 0.5 +
            results['grammar_score'] * 0.2,
            1
        )
        results['feedback'] = "Thank you for being honest. It's better to admit when you don't know something rather than guess. Consider reviewing this topic to strengthen your understanding."
        return results
    
    # Step 1: Keyword Matching (30% weight)
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Extract keywords from model answer (excluding stopwords)
        stop_words = set(stopwords.words('english'))
        model_words = set(word.lower() for word in word_tokenize(model_answer) 
                          if word.isalnum() and word.lower() not in stop_words)
        candidate_words = set(word.lower() for word in word_tokenize(candidate_answer) 
                              if word.isalnum() and word.lower() not in stop_words)
        
        if model_words:
            matched_keywords = model_words.intersection(candidate_words)
            keyword_score = (len(matched_keywords) / len(model_words)) * 100
            # More lenient: give minimum 20% credit if any keywords match
            if len(matched_keywords) > 0:
                keyword_score = max(keyword_score, 20)
            results['keyword_score'] = round(keyword_score, 1)
        else:
            results['keyword_score'] = 50  # Default if no keywords found
    except (ImportError, Exception) as e:
        # Fallback: simple word matching
        model_lower = model_answer.lower()
        candidate_lower = candidate_answer.lower()
        model_words = set(w for w in model_lower.split() if len(w) > 3)
        candidate_words = set(w for w in candidate_lower.split() if len(w) > 3)
        if model_words:
            matched = model_words.intersection(candidate_words)
            keyword_score = round((len(matched) / len(model_words)) * 100, 1)
            # More lenient: give minimum 20% credit if any keywords match
            if len(matched) > 0:
                keyword_score = max(keyword_score, 20)
            results['keyword_score'] = keyword_score
        else:
            results['keyword_score'] = 50
    
    # Step 2: Semantic Similarity (50% weight)
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
        
        # Load model (cache it for performance)
        if not hasattr(evaluate_answer_with_nlp, '_model'):
            evaluate_answer_with_nlp._model = SentenceTransformer('all-MiniLM-L6-v2')
        
        model = evaluate_answer_with_nlp._model
        
        # Encode sentences
        embeddings = model.encode([model_answer, candidate_answer], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        semantic_score = float(similarity) * 100
        # More lenient: give minimum 25% credit for any semantic similarity
        if semantic_score > 0:
            semantic_score = max(semantic_score, 25)
        results['semantic_score'] = round(semantic_score, 1)
    except (ImportError, Exception) as e:
        # Fallback: simple similarity based on common words
        model_words = set(model_answer.lower().split())
        candidate_words = set(candidate_answer.lower().split())
        if model_words or candidate_words:
            common = model_words.intersection(candidate_words)
            total = model_words.union(candidate_words)
            results['semantic_score'] = round((len(common) / max(1, len(total))) * 100, 1)
        else:
            results['semantic_score'] = 50
    
    # Step 3: Grammar Check (20% weight)
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(candidate_answer)
        grammar_score = max(0, 100 - len(matches) * 2)  # Deduct 2 points per error
        results['grammar_score'] = round(grammar_score, 1)
    except (ImportError, Exception) as e:
        # Fallback: basic grammar check (punctuation, capitalization)
        grammar_score = 100
        if candidate_answer and candidate_answer[0].islower():
            grammar_score -= 5
        if not candidate_answer.rstrip().endswith(('.', '!', '?')):
            grammar_score -= 5
        results['grammar_score'] = max(0, grammar_score)
    
    # Step 4: Combine scores with weights
    # Keyword: 30%, Semantic: 50%, Grammar: 20%
    final_score = (
        results['keyword_score'] * 0.3 +
        results['semantic_score'] * 0.5 +
        results['grammar_score'] * 0.2
    )
    results['final_score'] = round(final_score, 1)
    
    # Step 5: Generate feedback (more lenient and encouraging)
    if final_score >= 75:
        results['feedback'] = "Excellent! Your answer demonstrates good understanding of the concept."
    elif final_score >= 60:
        results['feedback'] = "Good answer! You've covered the main points well. Consider adding more examples or details to strengthen your response."
    elif final_score >= 45:
        results['feedback'] = "Fair answer. You're on the right track! Try to include more specific details or examples to improve your explanation."
    elif final_score >= 30:
        results['feedback'] = "Your answer shows some understanding of the topic. Review the key concepts and try to provide more comprehensive details."
    else:
        results['feedback'] = "Your answer needs more detail. Consider reviewing the concept and explaining it in your own words with examples."
    
    # Add specific feedback based on individual scores (more lenient thresholds)
    feedback_details = []
    if results['keyword_score'] < 40:
        feedback_details.append("Try to include more relevant key terms in your answer.")
    elif results['keyword_score'] < 60:
        feedback_details.append("You could include a few more key terms to strengthen your answer.")
    
    if results['semantic_score'] < 40:
        feedback_details.append("Your answer could better capture the main concept. Try explaining it more clearly.")
    elif results['semantic_score'] < 60:
        feedback_details.append("Your explanation is getting there - try to be more specific about the key points.")
    
    if results['grammar_score'] < 70:
        feedback_details.append("Pay attention to grammar and sentence structure to improve clarity.")
    
    if feedback_details:
        results['feedback'] += " " + " ".join(feedback_details)
    
    return results


@app.route('/')
def landing():
    """Landing page with Sign Up and Sign In buttons"""
    return render_template('landing.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Sign Up page with form validation"""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        errors = []

        if not name:
            errors.append('Name is required')

        if not email:
            errors.append('Email is required')
        elif not validate_email(email):
            errors.append('Please enter a valid email address')

        if not password:
            errors.append('Password is required')
        else:
            is_valid, error_msg = validate_password(password)
            if not is_valid:
                errors.append(error_msg)

        if password != confirm_password:
            errors.append('Passwords do not match')

        # Check if email already exists
        if email and validate_email(email):
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                errors.append('Email already registered. Please sign in instead.')

        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('signup.html', name=name, email=email)

        # Create new user
        try:
            password_hash = generate_password_hash(password)
            new_user = User(name=name, email=email, password_hash=password_hash)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully.', 'success')
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_email'] = user.email
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'error')
            return render_template('signup.html', name=name, email=email)

    return render_template('signup.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    """Sign In page with authentication"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        # Validation
        errors = []

        if not email:
            errors.append('Email is required')
        elif not validate_email(email):
            errors.append('Please enter a valid email address')

        if not password:
            errors.append('Password is required')

        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('signin.html', email=email)

        # Authenticate user
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_email'] = user.email
            flash(f'Welcome , {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect email or password', 'error')
            return render_template('signin.html', email=email)

    return render_template('signin.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page - requires authentication"""
    if 'user_id' not in session:
        flash('Please sign in to access the dashboard', 'error')
        return redirect(url_for('signin'))
    
    user_id = session.get('user_id')
    user = db.session.get(User, user_id)
    
    if not user:
        session.clear()
        flash('User not found. Please sign in again.', 'error')
        return redirect(url_for('signin'))
    
    # Fetch user's interviews
    interviews = Interview.query.filter_by(user_id=user_id).order_by(Interview.created_at.desc()).all()
    
    # Calculate progress statistics
    total_interviews = len(interviews)
    total_questions = sum(interview.total_questions for interview in interviews) if interviews else 0
    total_correct = sum(interview.correct_answers for interview in interviews) if interviews else 0
    overall_accuracy = round((total_correct / total_questions * 100), 1) if total_questions > 0 else 0
    
    # Calculate average score
    avg_score = round(sum(interview.score for interview in interviews) / total_interviews, 1) if total_interviews > 0 else 0
    
    return render_template(
        'dashboard.html',
        user_name=user.name,
        user_email=user.email,
        interviews=interviews,
        total_interviews=total_interviews,
        overall_accuracy=overall_accuracy,
        avg_score=avg_score,
        total_questions=total_questions
    )


@app.route('/interview')
def interview():
    """Interview page with domain selection and question area"""
    if 'user_id' not in session:
        flash('Please sign in to start an interview', 'error')
        return redirect(url_for('signin'))

    domain = request.args.get('domain')
    return render_template('interview.html', domain=domain)


@app.route('/progress')
def progress():
    """User progress over time with aggregates for charts"""
    if 'user_id' not in session:
        flash('Please sign in to view progress', 'error')
        return redirect(url_for('signin'))

    user_id = session['user_id']
    interviews = Interview.query.filter_by(user_id=user_id).order_by(Interview.created_at.asc()).all()

    if not interviews:
        return render_template('progress.html',
                               has_data=False,
                               user_name=session.get('user_name'))

    # Build series for Chart.js
    # X axis by date (YYYY-MM-DD), accumulate counts and average accuracy per day
    from collections import defaultdict
    per_day_counts = defaultdict(int)
    per_day_accuracy_sum = defaultdict(float)
    per_day_accuracy_count = defaultdict(int)
    domain_counts = defaultdict(int)

    for iv in interviews:
        date_key = iv.created_at.strftime('%Y-%m-%d') if iv.created_at else 'unknown'
        per_day_counts[date_key] += 1
        domain_counts[iv.domain] += 1
        per_day_accuracy_sum[date_key] += (iv.accuracy_rate or 0)
        per_day_accuracy_count[date_key] += 1

    labels = sorted(per_day_counts.keys())
    completed_series = [per_day_counts[d] for d in labels]
    accuracy_series = [round(per_day_accuracy_sum[d] / max(1, per_day_accuracy_count[d]), 1) for d in labels]

    total_interviews = len(interviews)
    avg_accuracy = round(sum(iv.accuracy_rate for iv in interviews) / total_interviews, 1) if total_interviews else 0

    # Estimated time spent: assume ~1.5 minutes per question as a simple proxy
    total_questions = sum(iv.total_questions for iv in interviews)
    est_total_minutes = round(total_questions * 1.5)

    # Strengths/weaknesses by domain via average accuracy
    domain_accuracy_sum = defaultdict(float)
    domain_accuracy_count = defaultdict(int)
    for iv in interviews:
        domain_accuracy_sum[iv.domain] += iv.accuracy_rate
        domain_accuracy_count[iv.domain] += 1
    domain_avg_accuracy = {k: round(domain_accuracy_sum[k] / max(1, domain_accuracy_count[k]), 1) for k in domain_accuracy_sum}

    # Most attempted domain
    most_attempted_domain = max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else None

    return render_template('progress.html',
                           has_data=True,
                           user_name=session.get('user_name'),
                           labels=labels,
                           completed_series=completed_series,
                           accuracy_series=accuracy_series,
                           total_interviews=total_interviews,
                           avg_accuracy=avg_accuracy,
                           est_total_minutes=est_total_minutes,
                           domain_counts=dict(domain_counts),
                           domain_avg_accuracy=domain_avg_accuracy,
                           most_attempted_domain=most_attempted_domain)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please sign in to view your profile', 'error')
        return redirect(url_for('signin'))

    user = db.session.get(User, session['user_id'])
    if not user:
        session.clear()
        flash('User not found. Please sign in again.', 'error')
        return redirect(url_for('signin'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        errors = []
        if not name:
            errors.append('Name is required')
        if not email or not validate_email(email):
            errors.append('Please enter a valid email address')

        # If changing email, ensure unique
        if email and email != user.email:
            if User.query.filter_by(email=email).first():
                errors.append('Email already in use by another account')

        # Password change flow (optional)
        if any([current_password, new_password, confirm_password]):
            if not current_password or not check_password_hash(user.password_hash, current_password):
                errors.append('Current password is incorrect')
            if not new_password:
                errors.append('New password is required')
            else:
                ok, msg = validate_password(new_password)
                if not ok:
                    errors.append(msg)
            if new_password != confirm_password:
                errors.append('New password and confirmation do not match')

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template('profile.html', name=name or user.name, email=email or user.email)

        # Apply updates
        user.name = name
        if email:
            user.email = email
        if new_password:
            user.password_hash = generate_password_hash(new_password)

        db.session.commit()
        session['user_name'] = user.name
        session['user_email'] = user.email
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))

    return render_template('profile.html', name=user.name, email=user.email)


def _configure_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if genai and api_key:
        genai.configure(api_key=api_key)
        print("Gemini API key configured")
        return genai
    return None


def _generate_question_with_gemini(domain: str, prev_answer: str | None = None, difficulty: str | None = None, index: int | None = None, asked_questions: list = None) -> dict:
    """Generate a beginner-friendly, easy-level technical interview question using Gemini."""
    if asked_questions is None:
        asked_questions = []
    
    # Always use easy difficulty for beginner-friendly questions
    difficulty = 'easy'
    
    client = _configure_gemini()
    
    # Build comprehensive prompt following the guidelines
    prompt = f"""You are an assistant for generating beginner-friendly, easy-level technical interview questions for {domain.upper()}.

Generate ONE easy-level interview question in {domain}. Follow these rules strictly:

1. Difficulty: ONLY easy level. Avoid medium or hard questions.
2. Question Style: Clear, simple, and practical. Focus on basic syntax, operations, and simple concepts.
3. Avoid Repetition: Do NOT repeat these questions: {', '.join(asked_questions[:5]) if asked_questions else 'None'}"""
    
    if prev_answer:
        prompt += f"""
4. Context-sensitive: The previous answer was: "{prev_answer}". If it contains misconceptions, generate a related question to correct it. Otherwise, generate a follow-up question that builds on it."""
    
    prompt += f"""

Return a well-formed JSON object with these EXACT fields:
- "question_id": a unique identifier (e.g., "{domain[:2]}-easy-001")
- "domain": "{domain}"
- "difficulty": "easy"
- "question": the interview question (clear, practical, and easy to answer)
- "answer": a short, concise answer (can include a code snippet if appropriate)
- "explanation": 1-3 sentences explaining why the answer is correct
- "hints": Optional array with 0-3 hints (or empty array [])
- "tags": Optional array of 1-3 tags (e.g., ['lists', 'syntax'])
- "example_input": Optional. Small example input if relevant
- "example_output": Optional. Corresponding output if relevant

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, just the JSON object."""
    
    if client:
        import json as _json
        max_retries = 3
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                generation_config = {
                    'temperature': 0.2,  # Low temperature for deterministic output
                    'max_output_tokens': 400,  # Optimal response size
                }
                resp = model.generate_content(prompt, generation_config=generation_config)
                text = resp.text.strip()
                
                # Remove markdown code blocks if present
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                
                # Extract JSON
                if '{' in text and '}' in text:
                    json_str = text[text.find('{'):text.rfind('}')+1]
                    data = _json.loads(json_str)
                    
                    # Validate and extract required fields
                    result = {
                        'question': data.get('question', 'Question unavailable.'),
                        'answer': data.get('answer', 'Answer unavailable.')
                    }
                    
                    # Return the full structured data if available
                    if 'question_id' in data and 'domain' in data:
                        return result  # For now, return simplified format for compatibility
                    return result
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, fall through to fallback
                    pass
                continue
    # Fallback static question pools - only easy questions for beginner-friendly interviews
    pools = {
        'python': {
            'easy': [
                ('What are lists in Python? Give two operations.', 'Lists are mutable sequences supporting append, pop, indexing, slicing, etc.'),
                ('How do you create a virtual environment?', 'python -m venv venv and activate it; use pip to manage deps.'),
                ('What is a dictionary in Python?', 'A key-value data structure, mutable and unordered.'),
                ('Explain Python indentation.', 'Python uses indentation to define code blocks instead of braces.'),
                ('What is PEP 8?', 'Python Enhancement Proposal 8 - style guide for Python code.'),
                ('What is a variable in Python?', 'A named location in memory that stores a value.'),
                ('How do you print something in Python?', 'Use the print() function: print("Hello, World!")'),
                ('What is a string in Python?', 'A sequence of characters enclosed in quotes (single or double).'),
                ('What is the difference between == and = in Python?', '= is for assignment, == is for comparison.'),
                ('What is a function in Python?', 'A reusable block of code defined with the def keyword.'),
            ],
        },
        'java': {
            'easy': [
                ('What is a class in Java?', 'A blueprint for objects containing fields and methods.'),
                ('Explain the purpose of the main method.', 'Entry point: public static void main(String[] args).'),
                ('What is an object in Java?', 'An instance of a class with state and behavior.'),
                ('Explain Java access modifiers.', 'public, private, protected, and default (package-private).'),
                ('What is inheritance in Java?', 'Mechanism where a class inherits properties and methods from another class.'),
                ('What is a variable in Java?', 'A named memory location that stores data of a specific type.'),
                ('What is the difference between int and Integer in Java?', 'int is a primitive type, Integer is a wrapper class.'),
                ('How do you declare an array in Java?', 'int[] arr = new int[5]; or int[] arr = {1, 2, 3};'),
                ('What is a method in Java?', 'A function defined within a class that performs a specific task.'),
                ('What is the String class in Java?', 'A class that represents a sequence of characters and is immutable.'),
            ],
        },
        'sql': {
            'easy': [
                ('What is a primary key?', 'A unique identifier for table rows; not null and unique.'),
                ('Difference between WHERE and HAVING?', 'WHERE filters rows before grouping; HAVING filters groups after aggregation.'),
                ('What is a foreign key?', 'A column that references the primary key of another table.'),
                ('Explain SELECT statement.', 'Used to query data from database tables.'),
                ('What is normalization?', 'Process of organizing data to reduce redundancy and improve integrity.'),
                ('What is a table in SQL?', 'A collection of related data organized in rows and columns.'),
                ('What is the INSERT statement used for?', 'To add new rows of data into a table.'),
                ('What is the UPDATE statement used for?', 'To modify existing data in a table.'),
                ('What is the DELETE statement used for?', 'To remove rows from a table.'),
                ('What is a database?', 'An organized collection of structured data stored electronically.'),
            ],
        },
    }
    
    d = domain.lower()
    if d not in pools:
        return { 'question': 'What is a variable?', 'answer': 'A named location in memory that stores a value.' }
    
    # Always use easy difficulty for beginner-friendly questions
    difficulty = 'easy'
    
    # Get available questions for easy difficulty level
    available_questions = pools[d].get(difficulty, [])
    
    # Filter out already asked questions
    if asked_questions:
        available_questions = [q for q in available_questions if q[0] not in asked_questions]
    
    # If all questions have been asked, reset and use all questions
    if not available_questions:
        available_questions = pools[d].get(difficulty, [])
    
    # If still no questions, use a generic fallback
    if not available_questions:
        return { 'question': f'Explain the basics of {domain}.', 'answer': f'{domain} is a programming language/database system.' }
    
    import random
    q, a = random.choice(available_questions)
    return { 'question': q, 'answer': a }


@app.route('/api/question', methods=['POST'])
def api_question():
    if 'user_id' not in session:
        return jsonify({ 'error': 'unauthorized' }), 401

    payload = request.get_json(silent=True) or {}
    domain = (payload.get('domain') or '').strip().lower()
    prev_answer = payload.get('previousAnswer')
    question_id = payload.get('questionId')  # If updating existing question with answer
    start_new = payload.get('startNew', False)  # Explicit flag to start a new session
    
    if domain not in ['python', 'java', 'sql']:
        return jsonify({ 'error': 'invalid_domain' }), 400

    # Store previous answer if provided
    if prev_answer and question_id:
        attempt = db.session.get(QuestionAttempt, question_id)
        if attempt and attempt.session_id in [s.id for s in InterviewSession.query.filter_by(user_id=session['user_id']).all()]:
            attempt.user_answer = prev_answer
            
            # Evaluate answer using both NLP and Gemini, then take the maximum score
            if attempt.correct_answer:
                # Check if this is an "I don't know" response
                candidate_lower = prev_answer.lower().strip()
                dont_know_phrases = [
                    "i don't know", "i don't", "i do not know", "i do not", 
                    "don't know", "do not know", "not sure", "unsure",
                    "i'm not sure", "i am not sure", "no idea", "have no idea",
                    "unfamiliar", "not familiar", "haven't learned", "have not learned"
                ]
                is_honest_response = any(phrase in candidate_lower for phrase in dont_know_phrases)
                
                # Run both evaluation methods
                nlp_results = evaluate_answer_with_nlp(attempt.correct_answer, prev_answer)
                gemini_results = evaluate_answer_with_gemini(attempt.correct_answer, prev_answer)
                
                # Store NLP component scores (for detailed breakdown)
                attempt.keyword_score = nlp_results['keyword_score']
                attempt.semantic_score = nlp_results['semantic_score']
                attempt.grammar_score = nlp_results['grammar_score']
                
                # Compare scores from both methods and use the maximum
                nlp_score = nlp_results['final_score']
                gemini_score = gemini_results['score']
                max_score = max(nlp_score, gemini_score)
                
                # Use the feedback from the method that gave the higher score
                # This ensures we use the most favorable evaluation
                if gemini_score >= nlp_score:
                    final_feedback = gemini_results['feedback']
                else:
                    final_feedback = nlp_results['feedback']
                
                # Store the maximum score as the final score
                # This gives the candidate the benefit of the more lenient evaluation
                attempt.final_nlp_score = max_score
                attempt.nlp_feedback = final_feedback
                
                # Set is_correct based on maximum score (more lenient threshold: 45%)
                # BUT: "I don't know" responses should NOT be marked as correct
                # They get lenient feedback but are still considered incorrect
                if is_honest_response:
                    attempt.is_correct = False  # "I don't know" is not correct, but feedback is encouraging
                else:
                    attempt.is_correct = max_score >= 45  # Normal threshold for other answers
            else:
                # Fallback to simple check if no model answer
                attempt.is_correct = prev_answer.lower().strip() in attempt.correct_answer.lower() if attempt.correct_answer else None
            
            db.session.commit()

    # Create or continue session robustly
    session_id = session.get('active_session_id')
    interview_session = db.session.get(InterviewSession, session_id) if session_id else None
    
    # If explicitly starting new session, clear any existing incomplete session
    if start_new:
        if interview_session and interview_session.completed_at is None:
            interview_session.completed_at = datetime.now(timezone.utc)
            db.session.commit()
        interview_session = None
        if 'active_session_id' in session:
            session.pop('active_session_id', None)
    
    # If storing previous answer, we're continuing an existing session
    if prev_answer and question_id:
        # Continue existing session
        if not interview_session or interview_session.user_id != session['user_id']:
            interview_session = InterviewSession.query.filter_by(user_id=session['user_id'], domain=domain, completed_at=None).order_by(InterviewSession.started_at.desc()).first()
    else:
        # Starting new question - check if we should create new session or continue
        # If no previous answer and no question_id, this is likely a fresh start
        # Check if there's an incomplete session that's very recent (within last 5 minutes)
        # If older than that, start fresh
        if not interview_session or interview_session.user_id != session['user_id']:
            # Check for incomplete session in same domain
            interview_session = InterviewSession.query.filter_by(user_id=session['user_id'], domain=domain, completed_at=None).order_by(InterviewSession.started_at.desc()).first()
            
            # If session exists but is old (more than 5 minutes), mark it as abandoned and start fresh
            if interview_session:
                # Handle both naive and aware datetimes (for backward compatibility with old data)
                started_at = interview_session.started_at
                if started_at.tzinfo is None:
                    # Convert naive datetime to UTC-aware
                    started_at = started_at.replace(tzinfo=timezone.utc)
                time_diff = (datetime.now(timezone.utc) - started_at).total_seconds()
                if time_diff > 300:  # 5 minutes
                    interview_session.completed_at = datetime.now(timezone.utc)
                    db.session.commit()
                    interview_session = None
        
        # If no incomplete session or session is completed, create new one
        if not interview_session or (interview_session.completed_at is not None):
            # Create fresh session with current_index = 0
            interview_session = InterviewSession(user_id=session['user_id'], domain=domain, current_index=0)
            db.session.add(interview_session)
            db.session.commit()
    
    session['active_session_id'] = interview_session.id

    # Check if we've already answered all questions BEFORE creating a new one
    # Count only answered questions (those with user_answer)
    answered_count = QuestionAttempt.query.filter_by(session_id=interview_session.id).filter(QuestionAttempt.user_answer.isnot(None)).count()
    
    if answered_count >= interview_session.total_questions:
        # All questions have been answered, mark as complete
        interview_session.completed_at = datetime.now(timezone.utc)
        # Create Interview record from session
        correct_count = sum(1 for a in QuestionAttempt.query.filter_by(session_id=interview_session.id).all() if a.is_correct is True)
        total = interview_session.total_questions
        interview = Interview(
            user_id=interview_session.user_id,
            domain=interview_session.domain,
            score=round((correct_count / max(1, total)) * 100, 1),
            total_questions=total,
            correct_answers=correct_count,
            performance_summary=f"Completed {interview_session.domain} interview with {correct_count}/{total} correct answers."
        )
        db.session.add(interview)
        # Clear active session id to avoid duplicate sessions on next start
        if 'active_session_id' in session:
            session.pop('active_session_id', None)
        db.session.commit()
        
        # Return completion status (no new question)
        return jsonify({
            'questionId': None,
            'question': None,
            'progress': 1.0,
            'total': interview_session.total_questions,
            'index': interview_session.total_questions,
            'completed': True,
            'sessionId': interview_session.id,
        })

    # Get list of already asked questions in this session to avoid duplicates
    existing_attempts = QuestionAttempt.query.filter_by(session_id=interview_session.id).all()
    asked_questions = [attempt.question_text for attempt in existing_attempts]

    # Decide difficulty based on previous correctness
    last_attempt = QuestionAttempt.query.filter_by(session_id=interview_session.id).order_by(QuestionAttempt.created_at.desc()).first()
    difficulty = None
    if prev_answer is not None and last_attempt and last_attempt.is_correct is not None:
        difficulty = 'hard' if last_attempt.is_correct else 'easy'

    # Generate next question (avoid duplicates)
    # Use current_index for display (1-based), but pass 0-based for difficulty calculation
    data = _generate_question_with_gemini(domain, prev_answer, difficulty=difficulty, index=interview_session.current_index, asked_questions=asked_questions)
    question_text = data.get('question', 'Question unavailable.')
    correct_answer = data.get('answer')

    # Check if this exact question was already asked (safety check)
    if question_text in asked_questions:
        # Try generating again with updated asked list
        data = _generate_question_with_gemini(domain, prev_answer, difficulty=difficulty, index=interview_session.current_index, asked_questions=asked_questions)
        question_text = data.get('question', 'Question unavailable.')
        correct_answer = data.get('answer')

    attempt = QuestionAttempt(
        session_id=interview_session.id,
        question_text=question_text,
        correct_answer=correct_answer,
    )
    db.session.add(attempt)
    interview_session.current_index += 1
    
    db.session.commit()

    progress = min(interview_session.current_index / interview_session.total_questions, 1.0)
    # Return 1-based index for display (current_index is already incremented, so it's correct)
    display_index = interview_session.current_index
    return jsonify({
        'questionId': attempt.id,
        'question': question_text,
        'progress': progress,
        'total': interview_session.total_questions,
        'index': display_index,  # This is already 1-based after increment
        'completed': False,  # Not complete yet - user still needs to answer this question
        'sessionId': interview_session.id,
    })


@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    if 'user_id' not in session:
        return jsonify({ 'error': 'unauthorized' }), 401

    # Stub: accept text fallback if provided
    text = request.form.get('text')
    if text:
        return jsonify({ 'transcript': text })

    # If audio is uploaded, we would forward to a Speech-to-Text provider
    # For now, return a friendly message
    return jsonify({ 'error': 'not_implemented', 'message': 'Speech-to-text not configured.' }), 501


@app.route('/api/voice-analysis', methods=['POST'])
def api_voice_analysis():
    """Enhanced voice analysis with clarity, confidence, and tone feedback based on entire interview"""
    if 'user_id' not in session:
        return jsonify({ 'error': 'unauthorized' }), 401

    # Get transcript and session data
    payload = request.get_json(silent=True) or {}
    transcript = payload.get('transcript', '').strip()
    session_id = payload.get('session_id') or session.get('active_session_id')
    voice_used = payload.get('voice_used', True)  # Flag to indicate if voice was actually used
    
    # Get all attempts from the session for comprehensive analysis
    all_transcripts = []
    all_questions = []
    interview_performance = {'correct': 0, 'total': 0, 'avg_score': 0}
    
    if session_id:
        interview_session = db.session.get(InterviewSession, session_id)
        if interview_session:
            attempts = QuestionAttempt.query.filter_by(session_id=session_id).order_by(QuestionAttempt.created_at.asc()).all()
            for attempt in attempts:
                if attempt.user_answer:
                    all_transcripts.append(attempt.user_answer)
                    all_questions.append({
                        'question': attempt.question_text,
                        'answer': attempt.user_answer,
                        'correct': attempt.is_correct,
                        'score': attempt.final_nlp_score
                    })
                    interview_performance['total'] += 1
                    if attempt.is_correct:
                        interview_performance['correct'] += 1
                    if attempt.final_nlp_score is not None:
                        interview_performance['avg_score'] += attempt.final_nlp_score
            
            if interview_performance['total'] > 0:
                interview_performance['avg_score'] = interview_performance['avg_score'] / interview_performance['total']
                interview_performance['accuracy'] = (interview_performance['correct'] / interview_performance['total']) * 100
    
    # Use all transcripts for analysis, or just the current one if no session
    combined_transcript = ' '.join(all_transcripts) if all_transcripts else transcript
    
    # Detect if voice was actually used (check for typed vs spoken characteristics)
    # Typed text usually has: proper capitalization, punctuation, no filler words, consistent spacing
    # Spoken text usually has: filler words (um, uh), inconsistent capitalization, fewer punctuation marks
    is_typed_text = False
    if combined_transcript:
        # Check for characteristics of typed text
        has_proper_caps = any(c.isupper() for c in combined_transcript if c.isalpha())
        has_punctuation = any(c in '.,!?;:' for c in combined_transcript)
        has_filler_words = any(word in combined_transcript.lower() for word in ['um', 'uh', 'er', 'ah'])
        word_count = len(combined_transcript.split())
        
        # If text has proper formatting but no filler words, likely typed
        if (has_proper_caps or has_punctuation) and not has_filler_words and word_count > 5:
            is_typed_text = True
    
    # If voice_used flag is False or we detect typed text, adjust feedback
    if not voice_used or is_typed_text:
        clarity_feedback = "Voice analysis is not applicable as answers were typed rather than spoken. Consider using voice input for a more natural interview experience."
        confidence_feedback = "Voice analysis is not applicable as answers were typed rather than spoken."
        tone_feedback = "Voice analysis is not applicable as answers were typed rather than spoken."
        clarity_score = 0
        confidence_score = 0
        tone_score = 0
    else:
        if not combined_transcript or len(combined_transcript) < 10:
            return jsonify({ 'error': 'transcript_too_short', 'message': 'Transcript must be at least 10 characters' }), 400

        # Analyze transcript characteristics for feedback
        word_count = len(combined_transcript.split())
        avg_word_length = sum(len(w) for w in combined_transcript.split()) / max(1, word_count)
        has_pauses = '...' in combined_transcript or 'um' in combined_transcript.lower() or 'uh' in combined_transcript.lower()
        sentence_count = len([s for s in combined_transcript.split('.') if s.strip()])
        avg_answer_length = word_count / max(1, len(all_transcripts)) if all_transcripts else word_count
        
        # Clarity analysis - dynamic based on interview performance
        clarity_score = 0.8
        if has_pauses:
            clarity_score -= 0.2
        if avg_word_length < 3:
            clarity_score -= 0.1
        if avg_answer_length < 15:
            clarity_score -= 0.1
        
        # Dynamic clarity feedback based on performance
        if interview_performance.get('avg_score', 0) >= 70:
            clarity_feedback = "Your speech was clear and articulate throughout the interview."
        elif interview_performance.get('avg_score', 0) >= 50:
            clarity_feedback = "Your speech was generally clear, but try to reduce pauses and speak more fluidly."
        elif has_pauses:
            clarity_feedback = "Try to improve clarity by reducing filler words and speaking at a steady pace."
        else:
            clarity_feedback = "Consider speaking more clearly and using complete words."
        
        # Confidence analysis - dynamic based on interview performance
        confidence_score = 0.75
        if avg_answer_length < 20:
            confidence_score -= 0.15
        if 'maybe' in combined_transcript.lower() or 'i think' in combined_transcript.lower():
            confidence_score -= 0.1
        if 'i don\'t know' in combined_transcript.lower() or 'not sure' in combined_transcript.lower():
            confidence_score -= 0.1
        
        # Dynamic confidence feedback based on performance
        if interview_performance.get('accuracy', 0) >= 80:
            confidence_feedback = "You demonstrated strong confidence and knowledge throughout the interview."
        elif interview_performance.get('accuracy', 0) >= 60:
            confidence_feedback = "You sounded confident in most answers. Keep building on this foundation."
        elif avg_answer_length < 20:
            confidence_feedback = "Consider providing more detailed answers to demonstrate confidence."
        elif 'maybe' in combined_transcript.lower() or 'i think' in combined_transcript.lower():
            confidence_feedback = "Try to project more confidence by avoiding hedging language."
        else:
            confidence_feedback = "Consider speaking more assertively and avoiding uncertain language."
        
        # Tone analysis - dynamic based on interview performance
        tone_score = 0.8
        if '!' in combined_transcript:
            tone_score += 0.05
        if any(word in combined_transcript.lower() for word in ['sorry', 'apologize', 'excuse']):
            tone_score -= 0.1
        
        # Dynamic tone feedback based on performance
        if interview_performance.get('avg_score', 0) >= 70:
            tone_feedback = "Your tone was friendly and professional throughout the interview."
        elif any(word in combined_transcript.lower() for word in ['sorry', 'apologize', 'excuse']):
            tone_feedback = "Your tone was professional, but avoid over-apologizing."
        elif tone_score < 0.7:
            tone_feedback = "Try to sound more engaged and maintain a professional yet friendly tone."
        else:
            tone_feedback = "Your tone was appropriate and professional."
    
    # Generate overall interview feedback using Gemini
    overall_feedback = ""
    if session_id and all_questions:
        client = _configure_gemini()
        if client:
            try:
                # Prepare interview summary for Gemini
                questions_summary = "\n".join([
                    f"Q{i+1}: {q['question']}\nA{i+1}: {q['answer']}\nScore: {q['score']:.1f}% ({'Correct' if q['correct'] else 'Incorrect'})\n"
                    for i, q in enumerate(all_questions)
                ])
                
                prompt = f"""You are an expert technical interviewer providing overall feedback on a completed interview.

Interview Performance Summary:
- Total Questions: {interview_performance['total']}
- Correct Answers: {interview_performance['correct']}
- Accuracy: {interview_performance.get('accuracy', 0):.1f}%
- Average Score: {interview_performance.get('avg_score', 0):.1f}%

Questions and Answers:
{questions_summary}

Provide constructive overall feedback (2-3 sentences) that:
1. Acknowledges strengths shown in the interview
2. Identifies areas for improvement
3. Offers encouragement and next steps

Be encouraging but honest. Focus on learning and growth.

Return ONLY a JSON object with this format:
{{
    "overall_feedback": "<your feedback message here>"
}}

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks."""
                
                import json as _json
                model = genai.GenerativeModel('gemini-1.5-flash')
                generation_config = {
                    'temperature': 0.5,
                    'max_output_tokens': 300,
                }
                resp = model.generate_content(prompt, generation_config=generation_config)
                text = resp.text.strip()
                
                # Remove markdown code blocks if present
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                
                # Extract JSON
                if '{' in text and '}' in text:
                    json_str = text[text.find('{'):text.rfind('}')+1]
                    data = _json.loads(json_str)
                    overall_feedback = data.get('overall_feedback', '')
            except Exception as e:
                # Fallback feedback if Gemini fails
                if interview_performance.get('accuracy', 0) >= 80:
                    overall_feedback = "Excellent performance! You demonstrated strong understanding of the concepts. Keep practicing to maintain this level."
                elif interview_performance.get('accuracy', 0) >= 60:
                    overall_feedback = "Good work! You showed solid knowledge in most areas. Review the topics you struggled with to strengthen your skills."
                else:
                    overall_feedback = "You've completed the interview. Review the questions and answers to identify areas for improvement. Keep learning and practicing!"
    
    # Store analysis if session exists (store on last attempt)
    if session_id:
        interview_session = db.session.get(InterviewSession, session_id)
        if interview_session:
            latest_attempt = QuestionAttempt.query.filter_by(session_id=session_id).order_by(QuestionAttempt.created_at.desc()).first()
            if latest_attempt:
                latest_attempt.voice_clarity_score = clarity_score
                latest_attempt.voice_confidence_score = confidence_score
                latest_attempt.voice_tone_analysis = tone_feedback
                db.session.commit()
    
    return jsonify({
        'clarity': clarity_feedback,
        'confidence': confidence_feedback,
        'tone': tone_feedback,
        'clarity_score': round(clarity_score * 100, 1),
        'confidence_score': round(confidence_score * 100, 1),
        'tone_score': round(tone_score * 100, 1),
        'overall_feedback': overall_feedback
    })


@app.route('/generate-report')
def generate_report():
    """Enhanced PDF report with question/answer details and voice analysis"""
    if 'user_id' not in session:
        flash('Please sign in to generate a report', 'error')
        return redirect(url_for('signin'))

    user_id = session['user_id']
    user = db.session.get(User, user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('dashboard'))

    # Get specific session if provided, otherwise all interviews
    session_id = request.args.get('session_id')
    
    if session_id:
        interview_session = InterviewSession.query.filter_by(id=session_id, user_id=user_id).first()
        if not interview_session:
            flash('Interview session not found', 'error')
            return redirect(url_for('dashboard'))
        
        attempts = QuestionAttempt.query.filter_by(session_id=session_id).order_by(QuestionAttempt.created_at.asc()).all()
        interviews_data = [(interview_session, attempts)]
    else:
        interviews = Interview.query.filter_by(user_id=user_id).order_by(Interview.created_at.desc()).all()
        interviews_data = [(iv, []) for iv in interviews]

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    def new_page_if_needed(current_y, min_y=100):
        if current_y < min_y:
            p.showPage()
            return height - 72
        return current_y

    y = height - 72
    p.setFont('Helvetica-Bold', 16)
    p.drawString(72, y, 'Technical Interview Report')
    y -= 24
    p.setFont('Helvetica', 12)
    p.drawString(72, y, f'Name: {user.name}  |  Email: {user.email}')
    y -= 18
    p.drawString(72, y, f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    y -= 30

    if not interviews_data or (not session_id and not interviews_data[0][0]):
        p.drawString(72, y, 'No completed interviews found.')
    else:
        for interview_or_session, attempts in interviews_data:
            y = new_page_if_needed(y)
            
            if session_id:
                # Session-based report
                p.setFont('Helvetica-Bold', 12)
                p.drawString(72, y, f'Interview Session: {interview_or_session.domain.upper()}')
                y -= 18
                p.setFont('Helvetica', 10)
                p.drawString(72, y, f'Started: {interview_or_session.started_at.strftime("%Y-%m-%d %H:%M") if interview_or_session.started_at else "N/A"}')
                y -= 16
                
                if attempts:
                    p.setFont('Helvetica-Bold', 11)
                    p.drawString(72, y, 'Questions & Answers:')
                    y -= 16
                    p.setFont('Helvetica', 10)
                    for idx, attempt in enumerate(attempts, 1):
                        y = new_page_if_needed(y, 120)
                        p.setFont('Helvetica-Bold', 10)
                        p.drawString(72, y, f'Q{idx}: {attempt.question_text[:80]}...' if len(attempt.question_text) > 80 else f'Q{idx}: {attempt.question_text}')
                        y -= 14
                        p.setFont('Helvetica', 9)
                        if attempt.user_answer:
                            p.drawString(88, y, f'Your Answer: {attempt.user_answer[:100]}...' if len(attempt.user_answer) > 100 else f'Your Answer: {attempt.user_answer}')
                            y -= 12
                        if attempt.is_correct is not None:
                            status = '✓ Correct' if attempt.is_correct else '✗ Incorrect'
                            p.drawString(88, y, status)
                            y -= 12
                        # NLP Evaluation Scores
                        if attempt.final_nlp_score is not None:
                            p.setFont('Helvetica-Bold', 9)
                            p.drawString(88, y, f'NLP Score: {attempt.final_nlp_score:.1f}%')
                            y -= 12
                            p.setFont('Helvetica', 8)
                            p.drawString(88, y, f'  Keyword: {attempt.keyword_score:.1f}% | Semantic: {attempt.semantic_score:.1f}% | Grammar: {attempt.grammar_score:.1f}%')
                            y -= 12
                            if attempt.nlp_feedback:
                                p.setFont('Helvetica-Oblique', 8)
                                feedback_text = attempt.nlp_feedback[:100] + ('...' if len(attempt.nlp_feedback) > 100 else '')
                                p.drawString(88, y, f'Feedback: {feedback_text}')
                                y -= 12
                                p.setFont('Helvetica', 9)
                        # Voice Analysis
                        if attempt.voice_clarity_score is not None:
                            p.setFont('Helvetica', 9)
                            p.drawString(88, y, f'Voice: Clarity {attempt.voice_clarity_score:.0f}%, Confidence {attempt.voice_confidence_score:.0f}%')
                            y -= 12
                        if attempt.voice_tone_analysis:
                            p.setFont('Helvetica-Oblique', 9)
                            p.drawString(88, y, f'Tone: {attempt.voice_tone_analysis[:80]}...' if len(attempt.voice_tone_analysis) > 80 else f'Tone: {attempt.voice_tone_analysis}')
                            y -= 12
                            p.setFont('Helvetica', 9)
                        y -= 8
            else:
                # Summary report
                p.setFont('Helvetica-Bold', 12)
                p.drawString(72, y, f'{interview_or_session.domain} | Score: {interview_or_session.score}/100 | Accuracy: {interview_or_session.accuracy_rate}%')
                y -= 16
                if interview_or_session.performance_summary:
                    p.setFont('Helvetica-Oblique', 10)
                    text = interview_or_session.performance_summary[:120] + ('...' if len(interview_or_session.performance_summary) > 120 else '')
                    p.drawString(88, y, f'"{text}"')
                    y -= 14
                    p.setFont('Helvetica', 11)

    p.showPage()
    p.save()
    buffer.seek(0)
    filename = f'interview_report_{session_id or "all"}.pdf' if session_id else 'interview_report_all.pdf'
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """User settings page for notification preferences"""
    if 'user_id' not in session:
        flash('Please sign in to view settings', 'error')
        return redirect(url_for('signin'))

    user_id = session['user_id']
    user = db.session.get(User, user_id)
    if not user:
        session.clear()
        flash('User not found. Please sign in again.', 'error')
        return redirect(url_for('signin'))

    # Get or create settings
    user_settings = UserSettings.query.filter_by(user_id=user_id).first()
    if not user_settings:
        user_settings = UserSettings(user_id=user_id)
        db.session.add(user_settings)
        db.session.commit()

    if request.method == 'POST':
        email_notifications = request.form.get('email_notifications') == 'on'
        user_settings.email_notifications = email_notifications
        user_settings.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        flash('Settings updated successfully', 'success')
        return redirect(url_for('settings'))

    return render_template('settings.html', 
                          email_notifications=user_settings.email_notifications,
                          user_name=user.name)


@app.route('/interview-history')
def interview_history():
    """View and replay past interview sessions"""
    if 'user_id' not in session:
        flash('Please sign in to view interview history', 'error')
        return redirect(url_for('signin'))

    user_id = session['user_id']
    sessions = InterviewSession.query.filter_by(user_id=user_id).order_by(InterviewSession.started_at.desc()).all()
    
    # Enrich with attempt counts, completion status, and attempts data
    sessions_data = []
    for sess in sessions:
        attempts = QuestionAttempt.query.filter_by(session_id=sess.id).order_by(QuestionAttempt.created_at.asc()).all()
        correct_count = sum(1 for a in attempts if a.is_correct is True)
        video = InterviewVideo.query.filter_by(session_id=sess.id, user_id=user_id).first()
        sessions_data.append({
            'session': sess,
            'attempts': attempts,  # Include attempts for display
            'total_attempts': len(attempts),
            'correct_answers': correct_count,
            'has_voice_analysis': any(a.voice_clarity_score is not None for a in attempts),
            'has_video': video is not None
        })

    return render_template('interview_history.html',
                          sessions_data=sessions_data,
                          user_name=session.get('user_name'))


def analyze_video_emotions_deepface(video_path, sample_interval=2.0):
    """
    Analyze video for emotions using OpenCV and DeepFace.
    
    Args:
        video_path: Path to video file
        sample_interval: Extract frame every N seconds (default: 2.0)
    
    Returns:
        dict with emotion analysis results or None if analysis fails
    """
    try:
        import cv2
        from deepface import DeepFace
        import numpy as np
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default FPS if not available
        frame_interval = int(fps * sample_interval)  # Frames to skip
        
        # Emotion tracking
        emotion_counts = {
            'neutral': 0,
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'fear': 0,
            'disgust': 0,
            'surprise': 0
        }
        
        frame_count = 0
        analyzed_frames = 0
        faces_detected = 0
        emotion_confidences = []
        
        print(f"Starting emotion analysis for {video_path}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames at specified interval
            if frame_count % frame_interval != 0:
                continue
            
            analyzed_frames += 1
            
            try:
                # Analyze frame with DeepFace
                # enforce_detection=False to avoid crashes if no face found
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                # Handle both single dict and list of dicts
                if isinstance(result, list):
                    result = result[0]
                
                # Extract dominant emotion
                if 'dominant_emotion' in result:
                    dominant = result['dominant_emotion'].lower()
                    if dominant in emotion_counts:
                        emotion_counts[dominant] += 1
                        faces_detected += 1
                    
                    # Store confidence scores
                    if 'emotion' in result:
                        emotions = result['emotion']
                        max_confidence = max(emotions.values()) if emotions else 0
                        emotion_confidences.append(max_confidence)
                
            except Exception as e:
                # Face not detected or other error - skip this frame
                print(f"Frame {frame_count}: {str(e)}")
                continue
        
        cap.release()
        
        if analyzed_frames == 0 or faces_detected == 0:
            print("No faces detected in video")
            return None
        
        # Calculate emotion distribution (percentages)
        total_detections = sum(emotion_counts.values())
        if total_detections == 0:
            return None
        
        emotion_distribution = {
            'neutral': round((emotion_counts['neutral'] / total_detections) * 100, 2),
            'happy': round((emotion_counts['happy'] / total_detections) * 100, 2),
            'sad': round((emotion_counts['sad'] / total_detections) * 100, 2),
            'angry': round((emotion_counts['angry'] / total_detections) * 100, 2),
            'fear': round((emotion_counts['fear'] / total_detections) * 100, 2),
            'disgust': round((emotion_counts['disgust'] / total_detections) * 100, 2),
            'surprise': round((emotion_counts['surprise'] / total_detections) * 100, 2)
        }
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_distribution, key=emotion_distribution.get)
        
        # Calculate engagement score (0-100)
        engagement_score = calculate_engagement_score(emotion_distribution, emotion_counts)
        
        # Average confidence
        avg_confidence = round(np.mean(emotion_confidences), 2) if emotion_confidences else 0
        
        return {
            'status': 'success',
            'dominant_emotion': dominant_emotion,
            'engagement_score': round(engagement_score, 2),
            'emotion_distribution': emotion_distribution,
            'emotion_counts': emotion_counts,
            'frames_analyzed': analyzed_frames,
            'faces_detected': faces_detected,
            'average_confidence': avg_confidence,
            'message': 'Analysis completed successfully'
        }
        
    except ImportError as e:
        print(f"Required library not available: {e}")
        return None
    except Exception as e:
        print(f"Error analyzing video with DeepFace: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_engagement_score(emotion_distribution, emotion_counts):
    """
    Calculate engagement score based on emotion distribution.
    
    Higher score if:
    - Mix of positive emotions (happy, surprise)
    - Lower neutral percentage
    - Active emotional expression
    
    Lower score if:
    - Mostly neutral (>70%)
    - High negative emotions (sad, angry, fear)
    """
    neutral_ratio = emotion_distribution.get('neutral', 0) / 100.0
    happy_ratio = emotion_distribution.get('happy', 0) / 100.0
    surprise_ratio = emotion_distribution.get('surprise', 0) / 100.0
    sad_ratio = emotion_distribution.get('sad', 0) / 100.0
    angry_ratio = emotion_distribution.get('angry', 0) / 100.0
    fear_ratio = emotion_distribution.get('fear', 0) / 100.0
    
    # Base score from positive emotions
    positive_ratio = happy_ratio + surprise_ratio
    base_score = positive_ratio * 60  # Max 60 points from positive emotions
    
    # Penalty for high neutral (lack of expression)
    neutral_penalty = max(0, (neutral_ratio - 0.3) * 40)  # Penalty if neutral > 30%
    
    # Penalty for negative emotions
    negative_ratio = sad_ratio + angry_ratio + fear_ratio
    negative_penalty = negative_ratio * 30  # Penalty for negative emotions
    
    # Bonus for emotional variety (not just neutral)
    variety_bonus = (1 - neutral_ratio) * 20  # Bonus for showing emotions
    
    # Calculate final score
    engagement_score = base_score + variety_bonus - neutral_penalty - negative_penalty
    
    # Clamp to 0-100 range
    engagement_score = max(0, min(100, engagement_score))
    
    return engagement_score


def analyze_video_emotions(video_path):
    """
    Legacy function for backward compatibility - uses basic OpenCV face detection.
    For new implementations, use analyze_video_emotions_deepface().
    """
    try:
        import cv2
        import numpy as np
        
        # Try to load face cascade (basic face detection)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frame_count = 0
        face_detected_frames = 0
        total_confidence = 0.0
        
        # Sample frames (every 30 frames to avoid processing all)
        sample_rate = 30
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                face_detected_frames += 1
                # Simple heuristic: larger face = closer to camera = more engaged
                for (x, y, w, h) in faces:
                    face_size = w * h
                    # Normalize confidence (0-1 scale)
                    confidence = min(1.0, face_size / (frame.shape[0] * frame.shape[1] * 0.1))
                    total_confidence += confidence
        
        cap.release()
        
        if frame_count == 0:
            return None
        
        # Calculate engagement metrics
        face_detection_rate = face_detected_frames / max(1, frame_count // sample_rate)
        avg_confidence = total_confidence / max(1, face_detected_frames)
        
        # Engagement score (0-10 scale)
        engagement_score = (face_detection_rate * 5) + (avg_confidence * 5)
        
        # Generate feedback
        feedback = []
        if face_detection_rate > 0.8:
            feedback.append("Excellent eye contact maintained throughout.")
        elif face_detection_rate > 0.6:
            feedback.append("Good eye contact, but could be more consistent.")
        else:
            feedback.append("Try to maintain better eye contact with the camera.")
        
        if avg_confidence > 0.7:
            feedback.append("You appeared engaged and attentive.")
        elif avg_confidence > 0.5:
            feedback.append("Consider sitting closer to the camera for better presence.")
        else:
            feedback.append("Try to position yourself better in the frame.")
        
        return {
            'engagement_score': round(engagement_score, 1),
            'face_detection_rate': round(face_detection_rate * 100, 1),
            'average_confidence': round(avg_confidence * 100, 1),
            'feedback': feedback,
            'frames_analyzed': frame_count // sample_rate,
            'faces_detected': face_detected_frames
        }
    except ImportError:
        # OpenCV not available
        return None
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return None


@app.route('/api/upload-video', methods=['POST'], endpoint='upload_video')
def upload_video():
    """Handle video upload from interview session"""
    if 'user_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    
    if 'video' not in request.files:
        return jsonify({'error': 'no_video_file'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'no_file_selected'}), 400
    
    # Get session_id and domain from form data
    session_id = request.form.get('session_id', type=int)
    domain = request.form.get('domain', '').strip()
    duration = request.form.get('duration', type=float)
    
    if not domain:
        return jsonify({'error': 'domain_required'}), 400
    
    # Validate session belongs to user
    if session_id:
        interview_session = InterviewSession.query.filter_by(id=session_id, user_id=session['user_id']).first()
        if not interview_session:
            return jsonify({'error': 'invalid_session'}), 400
    else:
        interview_session = None
    
    # Generate secure filename
    user_id = session['user_id']
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    original_filename = secure_filename(video_file.filename)
    file_ext = os.path.splitext(original_filename)[1] or '.webm'
    filename = f"user_{user_id}_session_{session_id or 'new'}_{timestamp}{file_ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save file
    try:
        video_file.save(filepath)
        file_size = os.path.getsize(filepath)
    except Exception as e:
        return jsonify({'error': 'save_failed', 'message': str(e)}), 500
    
    # Create database record first
    video_record = InterviewVideo(
        user_id=user_id,
        session_id=session_id,
        domain=domain,
        filename=filename,
        filepath=filepath,
        duration=duration,
        file_size=file_size
    )
    db.session.add(video_record)
    db.session.commit()
    
    # Analyze video for emotions/engagement using DeepFace (async-like, non-blocking)
    # Run analysis in background to avoid blocking the response
    try:
        # Try DeepFace analysis first
        analysis_result = analyze_video_emotions_deepface(filepath, sample_interval=2.0)
        
        if analysis_result and analysis_result.get('status') == 'success':
            # Store in EmotionAnalysis table
            emotion_analysis = EmotionAnalysis(
                user_id=user_id,
                video_id=video_record.id,
                session_id=session_id,
                neutral=analysis_result['emotion_distribution'].get('neutral', 0),
                happy=analysis_result['emotion_distribution'].get('happy', 0),
                sad=analysis_result['emotion_distribution'].get('sad', 0),
                angry=analysis_result['emotion_distribution'].get('angry', 0),
                fear=analysis_result['emotion_distribution'].get('fear', 0),
                disgust=analysis_result['emotion_distribution'].get('disgust', 0),
                surprise=analysis_result['emotion_distribution'].get('surprise', 0),
                engagement_score=analysis_result['engagement_score'],
                dominant_emotion=analysis_result['dominant_emotion'],
                frames_analyzed=analysis_result.get('frames_analyzed', 0),
                faces_detected=analysis_result.get('faces_detected', 0)
            )
            db.session.add(emotion_analysis)
            
            # Also update video record with summary
            video_record.emotion_analysis = json.dumps(analysis_result)
            video_record.engagement_score = analysis_result['engagement_score']
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'video_id': video_record.id,
                'filename': filename,
                'message': 'Video uploaded and analyzed successfully',
                'emotion_analysis': analysis_result
            })
        else:
            # Fallback to basic OpenCV analysis if DeepFace fails
            basic_analysis = analyze_video_emotions(filepath)
            if basic_analysis:
                video_record.emotion_analysis = json.dumps(basic_analysis)
                video_record.engagement_score = basic_analysis.get('engagement_score')
                db.session.commit()
    except Exception as e:
        print(f"Video analysis failed (non-critical): {e}")
        import traceback
        traceback.print_exc()
    
    return jsonify({
        'success': True,
        'video_id': video_record.id,
        'filename': filename,
        'message': 'Video uploaded successfully'
    })


@app.route('/api/analyze-emotion', methods=['POST'])
def analyze_emotion():
    """Analyze video for emotions using DeepFace"""
    if 'user_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    
    # Get video_id or video file
    video_id = request.form.get('video_id', type=int)
    video_file = request.files.get('video')
    
    if not video_id and not video_file:
        return jsonify({'error': 'video_id or video file required'}), 400
    
    # If video_id provided, use existing video
    if video_id:
        video = InterviewVideo.query.filter_by(id=video_id, user_id=session['user_id']).first()
        if not video:
            return jsonify({'error': 'video_not_found'}), 404
        video_path = video.filepath
    else:
        # Save uploaded file temporarily
        temp_filename = secure_filename(video_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{temp_filename}")
        video_file.save(temp_path)
        video_path = temp_path
    
    # Analyze video
    try:
        analysis_result = analyze_video_emotions_deepface(video_path, sample_interval=2.0)
        
        if not analysis_result or analysis_result.get('status') != 'success':
            return jsonify({
                'error': 'analysis_failed',
                'message': 'Could not analyze video. Ensure video contains visible faces.'
            }), 400
        
        # Store results if video_id was provided
        if video_id:
            # Check if analysis already exists
            existing = EmotionAnalysis.query.filter_by(video_id=video_id).first()
            if existing:
                # Update existing record
                existing.neutral = analysis_result['emotion_distribution'].get('neutral', 0)
                existing.happy = analysis_result['emotion_distribution'].get('happy', 0)
                existing.sad = analysis_result['emotion_distribution'].get('sad', 0)
                existing.angry = analysis_result['emotion_distribution'].get('angry', 0)
                existing.fear = analysis_result['emotion_distribution'].get('fear', 0)
                existing.disgust = analysis_result['emotion_distribution'].get('disgust', 0)
                existing.surprise = analysis_result['emotion_distribution'].get('surprise', 0)
                existing.engagement_score = analysis_result['engagement_score']
                existing.dominant_emotion = analysis_result['dominant_emotion']
                existing.frames_analyzed = analysis_result.get('frames_analyzed', 0)
                existing.faces_detected = analysis_result.get('faces_detected', 0)
            else:
                # Create new record
                emotion_analysis = EmotionAnalysis(
                    user_id=session['user_id'],
                    video_id=video_id,
                    session_id=video.session_id,
                    neutral=analysis_result['emotion_distribution'].get('neutral', 0),
                    happy=analysis_result['emotion_distribution'].get('happy', 0),
                    sad=analysis_result['emotion_distribution'].get('sad', 0),
                    angry=analysis_result['emotion_distribution'].get('angry', 0),
                    fear=analysis_result['emotion_distribution'].get('fear', 0),
                    disgust=analysis_result['emotion_distribution'].get('disgust', 0),
                    surprise=analysis_result['emotion_distribution'].get('surprise', 0),
                    engagement_score=analysis_result['engagement_score'],
                    dominant_emotion=analysis_result['dominant_emotion'],
                    frames_analyzed=analysis_result.get('frames_analyzed', 0),
                    faces_detected=analysis_result.get('faces_detected', 0)
                )
                db.session.add(emotion_analysis)
            
            # Update video record
            video.emotion_analysis = json.dumps(analysis_result)
            video.engagement_score = analysis_result['engagement_score']
            db.session.commit()
        
        # Clean up temp file if created
        if not video_id and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp file
        if not video_id and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        return jsonify({
            'error': 'analysis_error',
            'message': str(e)
        }), 500


@app.route('/api/video/<int:video_id>')
def get_video(video_id):
    """Serve video file"""
    if 'user_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    
    video = InterviewVideo.query.filter_by(id=video_id, user_id=session['user_id']).first()
    if not video:
        return jsonify({'error': 'video_not_found'}), 404
    
    return send_from_directory(
        os.path.dirname(video.filepath),
        os.path.basename(video.filepath),
        mimetype='video/webm'
    )


@app.route('/interview-replay/<int:session_id>')
def interview_replay(session_id):
    """Replay a specific interview session"""
    if 'user_id' not in session:
        flash('Please sign in to replay interviews', 'error')
        return redirect(url_for('signin'))

    interview_session = InterviewSession.query.filter_by(id=session_id, user_id=session['user_id']).first()
    if not interview_session:
        flash('Interview session not found', 'error')
        return redirect(url_for('interview_history'))

    attempts = QuestionAttempt.query.filter_by(session_id=session_id).order_by(QuestionAttempt.created_at.asc()).all()
    
    # Get video for this session
    video = InterviewVideo.query.filter_by(session_id=session_id, user_id=session['user_id']).first()
    
    # Get emotion analysis if available
    emotion_analysis = None
    if video:
        emotion_analysis = EmotionAnalysis.query.filter_by(video_id=video.id).first()
    
    return render_template('interview_replay.html',
                          interview_session=interview_session,
                          attempts=attempts,
                          video=video,
                          emotion_analysis=emotion_analysis,
                          user_name=session.get('user_name'))


@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('landing'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

