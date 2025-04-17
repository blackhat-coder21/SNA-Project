import pandas as pd
import random
from faker import Faker
import datetime as dt
import os

# Initialize faker
fake = Faker()
random.seed(42)
Faker.seed(42)

def generate_sample_data_files(num_users=50, output_dir="sample_data"):
    """
    Generate sample CSV files for social media, telecom, and incident reports.
    
    Parameters:
    num_users (int): Number of users to generate
    output_dir (str): Directory to save the CSV files
    
    Returns:
    tuple: Paths to the generated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate users
    users = []
    for i in range(num_users):
        name = fake.name()
        user_id = f"user_{i+1}"
        users.append({"id": user_id, "name": name})
    
    # Save users to CSV
    users_df = pd.DataFrame(users)
    users_path = os.path.join(output_dir, "users.csv")
    users_df.to_csv(users_path, index=False)
    print(f"Generated {users_path} with {len(users_df)} users")
    
    # Helper: get fuzzy variants of names to simulate inconsistent naming
    def name_variant(name):
        if random.random() < 0.3:
            parts = name.split()
            if len(parts) > 1:
                return f"{parts[0][0]}. {parts[-1]}"  # e.g., J. Smith
        elif random.random() < 0.5:
            return name.replace("e", "a") if "e" in name else name  # small typo
        return name
    
    # 1. Generate social media data
    social_data = []
    for _ in range(100):
        sender = random.choice(users)
        receiver = random.choice([u for u in users if u != sender])
        timestamp = fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
        message = fake.sentence(nb_words=6)
        platform = random.choice(["Twitter", "Facebook", "Instagram", "LinkedIn", "TikTok"])
        is_public = random.choice([True, False])
        likes = random.randint(0, 1000) if is_public else 0
        
        social_data.append([
            sender["id"], 
            name_variant(sender["name"]),
            receiver["id"], 
            name_variant(receiver["name"]),
            timestamp, 
            message,
            platform,
            is_public,
            likes
        ])
    
    social_df = pd.DataFrame(
        social_data, 
        columns=[
            "source", "source_name", "target", "target_name", 
            "timestamp", "message", "platform", "is_public", "likes"
        ]
    )
    
    social_path = os.path.join(output_dir, "social_media.csv")
    social_df.to_csv(social_path, index=False)
    print(f"Generated {social_path} with {len(social_df)} interactions")
    
    # 2. Generate telecom logs
    telecom_data = []
    for _ in range(150):
        caller = random.choice(users)
        receiver = random.choice([u for u in users if u != caller])
        duration = random.randint(10, 600)  # 10 seconds to 10 minutes
        timestamp = fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
        call_type = random.choice(["voice", "sms", "video"])
        status = random.choice(["completed", "missed", "rejected", "busy"])
        location = fake.city() if random.random() > 0.3 else None
        
        telecom_data.append([
            caller["id"], 
            name_variant(caller["name"]),
            receiver["id"], 
            name_variant(receiver["name"]),
            duration, 
            timestamp, 
            call_type,
            status,
            location
        ])
    
    telecom_df = pd.DataFrame(
        telecom_data, 
        columns=[
            "source", "source_name", "target", "target_name",
            "duration", "timestamp", "call_type", "status", "location"
        ]
    )
    
    telecom_path = os.path.join(output_dir, "telecom_logs.csv")
    telecom_df.to_csv(telecom_path, index=False)
    print(f"Generated {telecom_path} with {len(telecom_df)} call records")
    
    # 3. Generate incident reports
    incident_data = []
    for _ in range(50):
        reporter = random.choice(users)
        suspect = random.choice([u for u in users if u != reporter])
        incident_type = random.choice(["hoax call", "spam", "threat", "suspicious"])
        location = fake.city()
        timestamp = fake.date_time_between(start_date='-1y', end_date='now').strftime('%Y-%m-%d %H:%M:%S')
        severity = random.randint(1, 5)  # 1-5 severity scale
        resolved = random.choice([True, False])
        report_id = f"INC-{fake.unique.random_number(digits=6)}"
        
        incident_data.append([
            reporter["id"], 
            name_variant(reporter["name"]),
            suspect["id"], 
            name_variant(suspect["name"]),
            incident_type, 
            location, 
            timestamp,
            severity,
            resolved,
            report_id
        ])
    
    incident_df = pd.DataFrame(
        incident_data, 
        columns=[
            "source", "source_name", "target", "target_name",
            "incident_type", "location", "timestamp", "severity", "resolved", "report_id"
        ]
    )
    
    incident_path = os.path.join(output_dir, "incident_reports.csv")
    incident_df.to_csv(incident_path, index=False)
    print(f"Generated {incident_path} with {len(incident_df)} incident reports")
    
    return users_path, social_path, telecom_path, incident_path

if __name__ == "__main__":
    # Generate sample data with 50 users
    users_path, social_path, telecom_path, incident_path = generate_sample_data_files(num_users=50)
    
    print("\nAll CSV files have been saved to the 'sample_data' directory.")
    print("\nCSV files are ready for upload to the Social Network Analysis Dashboard.")