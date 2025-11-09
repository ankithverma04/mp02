"""
Helper script to add sample interview data for testing the dashboard.
Run this after creating a user account to populate the dashboard with sample data.
"""
from app import app, db, User, Interview
from datetime import datetime, timedelta

def add_sample_interviews(user_email):
    """Add sample interview data for a user"""
    with app.app_context():
        user = User.query.filter_by(email=user_email).first()
        
        if not user:
            print(f"User with email {user_email} not found!")
            print("Available users:")
            for u in User.query.all():
                print(f"  - {u.email}")
            return
        
        # Clear existing interviews for this user
        Interview.query.filter_by(user_id=user.id).delete()
        
        # Sample interviews
        sample_interviews = [
            {
                'domain': 'Python',
                'score': 85.0,
                'total_questions': 10,
                'correct_answers': 8,
                'performance_summary': 'Good understanding of data structures and algorithms. Strong problem-solving skills.',
                'created_at': datetime.utcnow() - timedelta(days=2)
            },
            {
                'domain': 'Java',
                'score': 78.5,
                'total_questions': 12,
                'correct_answers': 9,
                'performance_summary': 'Solid grasp of OOP concepts. Could improve on time complexity optimization.',
                'created_at': datetime.utcnow() - timedelta(days=5)
            },
            {
                'domain': 'SQL',
                'score': 92.0,
                'total_questions': 8,
                'correct_answers': 7,
                'performance_summary': 'Excellent database query skills. Strong understanding of joins and aggregations.',
                'created_at': datetime.utcnow() - timedelta(days=7)
            },
            {
                'domain': 'Python',
                'score': 88.0,
                'total_questions': 10,
                'correct_answers': 9,
                'performance_summary': 'Improved performance from previous attempt. Great work on recursion problems.',
                'created_at': datetime.utcnow() - timedelta(days=1)
            },
            {
                'domain': 'SQL',
                'score': 75.0,
                'total_questions': 10,
                'correct_answers': 7,
                'performance_summary': 'Good foundational knowledge. Focus on window functions and subqueries.',
                'created_at': datetime.utcnow() - timedelta(days=10)
            }
        ]
        
        for interview_data in sample_interviews:
            interview = Interview(
                user_id=user.id,
                **interview_data
            )
            db.session.add(interview)
        
        db.session.commit()
        print(f"Added {len(sample_interviews)} sample interviews for {user.name} ({user.email})")
        print(f"Total interviews: {len(sample_interviews)}")
        print(f"Average score: {sum(i['score'] for i in sample_interviews) / len(sample_interviews):.1f}/100")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        user_email = sys.argv[1]
    else:
        user_email = input("Enter user email to add sample interviews: ").strip()
    
    add_sample_interviews(user_email)


