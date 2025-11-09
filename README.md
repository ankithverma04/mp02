# Technical Interview Simulation Platform

A modern web application for technical interview practice with user authentication, light/dark mode support, and a clean, responsive design.

## Features

- **Landing Page**: Welcome page with Sign Up and Sign In buttons
- **User Authentication**: Secure sign up and sign in with password hashing
- **Dashboard**: Comprehensive dashboard showing user progress, completed interviews, and statistics
- **Interview**: Domain selection, question flow powered by Gemini (stub fallback), voice capture, waveform, and voice feedback
- **Progress**: Charts for progress over time, domain distribution, KPIs (Chart.js)
- **Profile**: Update name, email, and change password with validation
- **Theme Toggle**: Light and dark mode with localStorage persistence
- **Form Validation**: Frontend and backend validation for all forms
- **Responsive Design**: Mobile-first design using Tailwind CSS
- **Accessibility**: WCAG AA compliant with ARIA labels

## Tech Stack

- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Backend**: Flask (Python)
- **Database**: SQLite (default) or PostgreSQL/MySQL
- **Password Hashing**: Werkzeug (bcrypt)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):
Create a `.env` file:
```
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///interview_platform.db
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
mp01/
├── app.py                 # Flask application and routes
├── requirements.txt       # Python dependencies
├── templates/
│   ├── base.html         # Base template with theme toggle
│   ├── landing.html      # Landing page
│   ├── signup.html      # Sign up page
│   ├── signin.html      # Sign in page
│   ├── dashboard.html   # Dashboard page
│   ├── progress.html    # Progress analytics page
│   └── profile.html     # Profile page
│   └── interview.html   # Interview page
└── README.md
```

## Usage

### Landing Page
- Visit the root URL to see the landing page
- Use the theme toggle button (top right) to switch between light and dark modes
- Click "Sign Up" or "Sign In" to proceed

### Sign Up
- Enter your name, email, password, and confirm password
- Password must be at least 8 characters with one number and one special character
- Email must be in a valid format
- Passwords must match

### Sign In
- Enter your email and password
- You'll be redirected to the dashboard upon successful login

### Dashboard
- View your progress statistics (total interviews, overall accuracy, average score)
- See a list of all completed interviews with performance summaries
- Each interview card shows:
  - Domain (Python, Java, SQL, etc.)
  - Score and accuracy rate
  - Performance feedback
  - Date and time completed
- Click "Start New Interview" to begin a new interview session
- Empty state message appears if no interviews have been completed yet

## Design Specifications

### Colors
- **Background**: White (Light) / Black (Dark)
- **Primary Button**: Bright Blue (hsl(203.89, 88.28%, 53.14%))
- **Text**: Dark Gray (Light) / Light Gray (Dark)
- **Card**: Light Gray (Light) / Dark Gray (Dark)
- **Input Field**: Light Gray (hsl(200, 23.08%, 97.45%))

### Fonts
- **Primary**: Open Sans (sans-serif)
- **Headings**: Georgia (serif)

### Layout
- Centered content using Tailwind CSS grid
- Rounded corners (1.3rem) for buttons and inputs
- Mobile-first responsive design

## Database

The application uses SQLite by default. To use PostgreSQL or MySQL, update the `DATABASE_URL` in your `.env` file:

```
DATABASE_URL=postgresql://user:password@localhost/dbname
# or
DATABASE_URL=mysql://user:password@localhost/dbname
```

### Use PostgreSQL on Neon (serverless)

1) Create a Neon project and a database. Copy the connection string from the Neon dashboard (ensure SSL required).

2) Set the `DATABASE_URL` in your `.env` to the Neon URL. Use the SQLAlchemy URI with psycopg2 and require SSL, e.g.:

```
DATABASE_URL=postgresql+psycopg2://neondb_owner:YOUR_PASSWORD@YOUR_HOST/neondb?sslmode=require
```

3) Install dependencies:

```bash
pip install -r requirements.txt
```

4) Initialize migrations (first time only):

```bash
set FLASK_APP=app.py  # Windows PowerShell: $env:FLASK_APP="app.py"
flask db init
flask db migrate -m "init schema"
flask db upgrade
```

5) Run the app normally:

```bash
python app.py
```

### Migrate existing SQLite data to Neon

1) Ensure your Postgres schema is created (run migrations as above).

2) Set the following env vars:

```
SQLITE_URL=sqlite:///interview_platform.db
POSTGRES_URL=postgresql+psycopg2://neondb_owner:YOUR_PASSWORD@YOUR_HOST/neondb?sslmode=require
```

3) Run the migration script:

```bash
python scripts/migrate_to_neon.py
```

The script copies Users, UserSettings, Interviews, InterviewSessions, and QuestionAttempts, skipping duplicates.

Data migration from SQLite (optional): If you have existing data in SQLite you want to move, you can export tables to CSV and import into Postgres, or temporarily point the app at SQLite, dump via a small script using SQLAlchemy ORM, and re-create objects in Postgres. For production migrations, consider a one-off ETL script.

## Security Notes

- Passwords are hashed using Werkzeug's password hashing
- Session management for user authentication
- CSRF protection (should be added for production)
- Input validation on both frontend and backend

## Sample Data

To add sample interview data for testing the dashboard:

```bash
python add_sample_data.py your-email@example.com
```

This will add 5 sample interviews across different domains (Python, Java, SQL) with various scores and performance summaries.

## Database Models

### User
- `id`: Primary key
- `name`: User's full name
- `email`: Unique email address
- `password_hash`: Hashed password
- `interviews`: Relationship to Interview records

### Interview
- `id`: Primary key
- `user_id`: Foreign key to User
- `domain`: Interview domain (Python, Java, SQL, etc.)
- `score`: Score out of 100
- `total_questions`: Total number of questions
- `correct_answers`: Number of correct answers
- `performance_summary`: Brief feedback text
- `created_at`: Timestamp of interview completion

### InterviewSession
- `id`, `user_id`, `domain`, `started_at`, `completed_at`, `current_index`, `total_questions`

### QuestionAttempt
- `id`, `session_id`, `question_text`, `correct_answer`, `user_answer`, `is_correct`, `created_at`

## Development

To run in development mode:
```bash
python app.py
```

The app will run in debug mode on `http://localhost:5000`

### Database Setup

The database will be automatically created when you first run the application. If you need to reset the database:

1. Delete the `interview_platform.db` file (for SQLite)
2. Restart the application - it will create a fresh database

