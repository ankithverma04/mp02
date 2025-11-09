from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import re
import os
from datetime import datetime
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
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 
    'sqlite:///interview_platform.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads', 'videos')

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
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

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
    started_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
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
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
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
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


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
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
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
            results['keyword_score'] = round((len(matched) / len(model_words)) * 100, 1)
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
    
    # Step 5: Generate feedback
    if final_score >= 85:
        results['feedback'] = "Excellent! Your answer is clear, accurate, and well-structured."
    elif final_score >= 70:
        results['feedback'] = "Good answer! You covered the main points, but could add more detail or examples."
    elif final_score >= 60:
        results['feedback'] = "Fair answer. You're on the right track, but missed some important concepts. Try to be more specific."
    elif final_score >= 50:
        results['feedback'] = "Needs improvement. Your answer touches on the topic but lacks key details. Review the concept and try again."
    else:
        results['feedback'] = "Incorrect or incomplete answer. Please review the concept and provide a more comprehensive explanation."
    
    # Add specific feedback based on individual scores
    feedback_details = []
    if results['keyword_score'] < 60:
        feedback_details.append("Consider including more key terms from the question.")
    if results['semantic_score'] < 60:
        feedback_details.append("Your answer doesn't fully capture the meaning. Try explaining the concept more clearly.")
    if results['grammar_score'] < 80:
        feedback_details.append("Pay attention to grammar and sentence structure.")
    
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
            flash('Account created successfully! Please sign in.', 'success')
            return redirect(url_for('signin'))
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
            flash(f'Welcome back, {user.name}!', 'success')
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
    user = User.query.get(user_id)
    
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

    user = User.query.get(session['user_id'])
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
        return genai
    return None


def _generate_question_with_gemini(domain: str, prev_answer: str | None = None, difficulty: str | None = None, index: int | None = None, asked_questions: list = None) -> dict:
    """Generate a question using Gemini (stubbed fallback if no API key)."""
    if asked_questions is None:
        asked_questions = []
    
    client = _configure_gemini()
    prompt = f"Generate a single {domain} technical interview question. Provide JSON with fields: question, answer."
    if prev_answer:
        prompt += f" Consider the previous answer: {prev_answer}. Adjust difficulty accordingly."
    if asked_questions:
        prompt += f" Avoid these questions already asked: {', '.join(asked_questions[:3])}."
    if client:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            if difficulty:
                prompt = f"{prompt} Difficulty: {difficulty}."
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            # Very naive parse fallback; ideally use a JSON prompt and parse safely
            if '{' in text and '}' in text:
                json_str = text[text.find('{'):text.rfind('}')+1]
                import json as _json
                data = _json.loads(json_str)
                return { 'question': data.get('question', 'Question unavailable.'), 'answer': data.get('answer') }
        except Exception:
            pass
    # Fallback static question pools per domain with simple difficulty bands
    pools = {
        'python': {
            'easy': [
                ('What are lists in Python? Give two operations.', 'Lists are mutable sequences supporting append, pop, indexing, slicing, etc.'),
                ('How do you create a virtual environment?', 'python -m venv venv and activate it; use pip to manage deps.'),
                ('What is a dictionary in Python?', 'A key-value data structure, mutable and unordered.'),
                ('Explain Python indentation.', 'Python uses indentation to define code blocks instead of braces.'),
                ('What is PEP 8?', 'Python Enhancement Proposal 8 - style guide for Python code.'),
            ],
            'medium': [
                ('Explain list vs tuple in Python with use-cases.', 'Tuples are immutable, lists are mutable; tuples for fixed records, lists for dynamic collections.'),
                ('What is a generator and yield? Provide an example use-case.', 'Generators lazily produce values with yield; used for streaming/large data.'),
                ('What are decorators in Python?', 'Functions that modify other functions, using @syntax.'),
                ('Explain *args and **kwargs.', '*args for variable positional arguments, **kwargs for keyword arguments.'),
                ('What is list comprehension?', 'Concise way to create lists: [x*2 for x in range(10)].'),
            ],
            'hard': [
                ('How does the GIL affect multi-threading in CPython?', 'GIL allows only one bytecode thread at a time; use multiprocessing or I/O-bound threads.'),
                ('When to use dataclasses vs attrs vs pydantic?', 'Dataclasses for simple data containers, attrs for more features, pydantic for validation.'),
                ('Explain Python\'s memory management and garbage collection.', 'Uses reference counting and cyclic garbage collector.'),
                ('What are metaclasses in Python?', 'Classes that create classes; used for advanced class customization.'),
                ('Explain context managers and the with statement.', 'Manages resources automatically; uses __enter__ and __exit__ methods.'),
            ],
        },
        'java': {
            'easy': [
                ('What is a class in Java?', 'A blueprint for objects containing fields and methods.'),
                ('Explain the purpose of the main method.', 'Entry point: public static void main(String[] args).'),
                ('What is an object in Java?', 'An instance of a class with state and behavior.'),
                ('Explain Java access modifiers.', 'public, private, protected, and default (package-private).'),
                ('What is inheritance in Java?', 'Mechanism where a class inherits properties and methods from another class.'),
            ],
            'medium': [
                ('Difference between interface and abstract class?', 'Interfaces define contracts; abstract classes can have state and common behavior.'),
                ('Explain JVM, JRE, JDK differences.', 'JVM runs bytecode; JRE = JVM + libs; JDK = JRE + tools.'),
                ('What is method overriding vs overloading?', 'Overriding: same signature in subclass; Overloading: same name, different parameters.'),
                ('Explain Java collections framework.', 'Set of classes and interfaces for storing and manipulating groups of objects.'),
                ('What are exceptions in Java?', 'Events that disrupt normal flow; handled with try-catch blocks.'),
            ],
            'hard': [
                ('Describe Java memory model and happens-before.', 'Defines concurrency rules; happens-before ensures visibility and ordering.'),
                ('Compare Parallel, CMS, and G1 collectors.', 'Different GC algorithms with trade-offs in pause times and throughput.'),
                ('Explain Java generics and type erasure.', 'Type parameters for classes/methods; erased at compile time for backward compatibility.'),
                ('What is the volatile keyword?', 'Ensures variable visibility across threads; prevents caching.'),
                ('Explain Java reflection API.', 'Runtime inspection and manipulation of classes, methods, and fields.'),
            ],
        },
        'sql': {
            'easy': [
                ('What is a primary key?', 'A unique identifier for table rows; not null and unique.'),
                ('Difference between WHERE and HAVING?', 'WHERE filters rows before grouping; HAVING filters groups after aggregation.'),
                ('What is a foreign key?', 'A column that references the primary key of another table.'),
                ('Explain SELECT statement.', 'Used to query data from database tables.'),
                ('What is normalization?', 'Process of organizing data to reduce redundancy and improve integrity.'),
            ],
            'medium': [
                ('Write SQL to find the second highest salary.', 'SELECT MAX(salary) FROM t WHERE salary < (SELECT MAX(salary) FROM t);'),
                ('Explain INNER vs LEFT JOIN with an example.', 'INNER returns matches; LEFT keeps all left rows with nulls for missing right.'),
                ('What is a subquery?', 'A query nested inside another query.'),
                ('Explain GROUP BY clause.', 'Groups rows with same values into summary rows.'),
                ('What is an index in SQL?', 'Database structure that improves speed of data retrieval.'),
            ],
            'hard': [
                ('When to use window functions? Example with ROW_NUMBER.', 'Use for analytics over partitions; SELECT ROW_NUMBER() OVER(PARTITION BY d ORDER BY x).'),
                ('Explain transaction isolation levels and anomalies.', 'Read uncommitted..serializable; prevents dirty/non-repeatable/phantom reads.'),
                ('What is ACID in database transactions?', 'Atomicity, Consistency, Isolation, Durability - transaction properties.'),
                ('Explain SQL query optimization techniques.', 'Indexing, query rewriting, execution plan analysis, and statistics.'),
                ('What are stored procedures?', 'Precompiled SQL statements stored in the database for reuse.'),
            ],
        },
    }
    d = domain.lower()
    if d not in pools:
        return { 'question': 'Describe Big-O of binary search.', 'answer': 'O(log n)' }
    # Choose difficulty based on provided difficulty or index progression
    if not difficulty and index is not None:
        if index <= 1:
            difficulty = 'easy'
        elif index <= 3:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
    difficulty = difficulty or 'medium'
    
    # Get available questions for this difficulty level
    available_questions = pools[d][difficulty]
    
    # Filter out already asked questions
    if asked_questions:
        available_questions = [q for q in available_questions if q[0] not in asked_questions]
    
    # If all questions have been asked, reset and use all questions
    if not available_questions:
        available_questions = pools[d][difficulty]
    
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
        attempt = QuestionAttempt.query.get(question_id)
        if attempt and attempt.session_id in [s.id for s in InterviewSession.query.filter_by(user_id=session['user_id']).all()]:
            attempt.user_answer = prev_answer
            
            # Evaluate answer using NLP
            if attempt.correct_answer:
                nlp_results = evaluate_answer_with_nlp(attempt.correct_answer, prev_answer)
                
                # Store NLP scores
                attempt.keyword_score = nlp_results['keyword_score']
                attempt.semantic_score = nlp_results['semantic_score']
                attempt.grammar_score = nlp_results['grammar_score']
                attempt.final_nlp_score = nlp_results['final_score']
                attempt.nlp_feedback = nlp_results['feedback']
                
                # Set is_correct based on final NLP score (threshold: 60%)
                attempt.is_correct = nlp_results['final_score'] >= 60
            else:
                # Fallback to simple check if no model answer
                attempt.is_correct = prev_answer.lower().strip() in attempt.correct_answer.lower() if attempt.correct_answer else None
            
            db.session.commit()

    # Create or continue session robustly
    session_id = session.get('active_session_id')
    interview_session = InterviewSession.query.get(session_id) if session_id else None
    
    # If explicitly starting new session, clear any existing incomplete session
    if start_new:
        if interview_session and interview_session.completed_at is None:
            interview_session.completed_at = datetime.utcnow()
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
                time_diff = (datetime.utcnow() - interview_session.started_at).total_seconds()
                if time_diff > 300:  # 5 minutes
                    interview_session.completed_at = datetime.utcnow()
                    db.session.commit()
                    interview_session = None
        
        # If no incomplete session or session is completed, create new one
        if not interview_session or (interview_session.completed_at is not None):
            # Create fresh session with current_index = 0
            interview_session = InterviewSession(user_id=session['user_id'], domain=domain, current_index=0)
            db.session.add(interview_session)
            db.session.commit()
    
    session['active_session_id'] = interview_session.id

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
    
    # Mark session as complete if reached total
    completed = False
    if interview_session.current_index >= interview_session.total_questions:
        interview_session.completed_at = datetime.utcnow()
        # Create Interview record from session
        correct_count = sum(1 for a in QuestionAttempt.query.filter_by(session_id=interview_session.id).all() if a.is_correct is True)
        total = interview_session.current_index
        interview = Interview(
            user_id=interview_session.user_id,
            domain=interview_session.domain,
            score=round((correct_count / max(1, total)) * 100, 1),
            total_questions=total,
            correct_answers=correct_count,
            performance_summary=f"Completed {interview_session.domain} interview with {correct_count}/{total} correct answers."
        )
        db.session.add(interview)
        completed = True
        # Clear active session id to avoid duplicate sessions on next start
        if 'active_session_id' in session:
            session.pop('active_session_id', None)
    
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
        'completed': completed,
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
    """Enhanced voice analysis with clarity, confidence, and tone feedback"""
    if 'user_id' not in session:
        return jsonify({ 'error': 'unauthorized' }), 401

    # Get transcript and session data
    payload = request.get_json(silent=True) or {}
    transcript = payload.get('transcript', '').strip()
    session_id = session.get('active_session_id')
    
    if not transcript or len(transcript) < 10:
        return jsonify({ 'error': 'transcript_too_short', 'message': 'Transcript must be at least 10 characters' }), 400

    # Analyze transcript characteristics for feedback
    word_count = len(transcript.split())
    avg_word_length = sum(len(w) for w in transcript.split()) / max(1, word_count)
    has_pauses = '...' in transcript or 'um' in transcript.lower() or 'uh' in transcript.lower()
    sentence_count = len([s for s in transcript.split('.') if s.strip()])
    
    # Clarity analysis
    clarity_score = 0.8
    clarity_feedback = "Your speech was clear and articulate."
    if has_pauses:
        clarity_score -= 0.2
        clarity_feedback = "Your speech was generally clear, but try to reduce pauses and speak more fluidly."
    if avg_word_length < 3:
        clarity_score -= 0.1
        clarity_feedback = "Consider speaking more clearly and using complete words."
    if clarity_score < 0.6:
        clarity_feedback = "Try to improve clarity by reducing filler words and speaking at a steady pace."
    
    # Confidence analysis (based on transcript characteristics)
    confidence_score = 0.75
    confidence_feedback = "You sounded confident and assertive."
    if word_count < 20:
        confidence_score -= 0.15
        confidence_feedback = "Consider providing more detailed answers to demonstrate confidence."
    if 'maybe' in transcript.lower() or 'i think' in transcript.lower():
        confidence_score -= 0.1
        confidence_feedback = "Try to project more confidence by avoiding hedging language."
    if confidence_score < 0.6:
        confidence_feedback = "Consider speaking more assertively and avoiding uncertain language."
    
    # Tone analysis
    tone_score = 0.8
    tone_feedback = "Your tone was friendly and professional."
    if '!' in transcript:
        tone_score += 0.05
    if any(word in transcript.lower() for word in ['sorry', 'apologize', 'excuse']):
        tone_score -= 0.1
        tone_feedback = "Your tone was professional, but avoid over-apologizing."
    if tone_score < 0.7:
        tone_feedback = "Try to sound more engaged and maintain a professional yet friendly tone."
    
    # Store analysis if session exists
    if session_id:
        interview_session = InterviewSession.query.get(session_id)
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
        'tone_score': round(tone_score * 100, 1)
    })


@app.route('/generate-report')
def generate_report():
    """Enhanced PDF report with question/answer details and voice analysis"""
    if 'user_id' not in session:
        flash('Please sign in to generate a report', 'error')
        return redirect(url_for('signin'))

    user_id = session['user_id']
    user = User.query.get(user_id)
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
    p.drawString(72, y, f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')
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
    user = User.query.get(user_id)
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
        user_settings.updated_at = datetime.utcnow()
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
    
    # Enrich with attempt counts and completion status
    sessions_data = []
    for sess in sessions:
        attempts = QuestionAttempt.query.filter_by(session_id=sess.id).all()
        correct_count = sum(1 for a in attempts if a.is_correct is True)
        video = InterviewVideo.query.filter_by(session_id=sess.id, user_id=user_id).first()
        sessions_data.append({
            'session': sess,
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
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
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
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{temp_filename}")
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
                          session=interview_session,
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

