import pandas as pd
import random

rows = []

for _ in range(500):
    attendance_percentage = random.uniform(40, 95)
    leave_percentage = 100 - attendance_percentage
    internal_marks = random.randint(35, 90)

    if attendance_percentage >= 75:
        status = "Safe"
    elif attendance_percentage >= 60:
        status = "At Risk"
    else:
        status = "Critical"

    rows.append([
        attendance_percentage,
        leave_percentage,
        internal_marks,
        status
    ])

df = pd.DataFrame(
    rows,
    columns=[
        "attendance_percentage",
        "leave_percentage",
        "internal_marks",
        "status"
    ]
)

df.to_csv("attendance_data.csv", index=False)

print("✅ 500-row percentage-based dataset generated")
