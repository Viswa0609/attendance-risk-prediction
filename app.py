import streamlit as st
import numpy as np
import joblib
import math

# Load trained ML model
model = joblib.load("attendance_model.pkl")

# ---------- GOAL-BASED EMOJI ----------
def attendance_emoji(att, goal):
    diff = att - goal
    if diff >= 0:
        return "😊 Safe"
    elif diff >= -5:
        return "🙂 Almost there"
    elif diff >= -10:
        return "😐 Slight Risk"
    elif diff >= -15:
        return "😕 At Risk"
    elif diff >= -20:
        return "😟 High Risk"
    elif diff >= -25:
        return "😰 Very High Risk"
    else:
        return "💀 Dead"

def format_ml_risk(risk):
    if risk == "Safe":
        return "😊 Safe"
    elif risk == "At Risk":
        return "😟 At Risk"
    elif risk == "Critical":
        return "💀 Critical"
    else:
        return risk

# ---------- UI ----------
st.title("🎓 Student Attendance Risk Prediction System")

st.header("📥 Enter Attendance Details")

classes_per_day = st.number_input("Classes per day", min_value=1, value=6)
total_working_days = st.number_input("Total working days in semester", min_value=1, value=80)
days_completed = st.number_input("Days completed so far", min_value=0, value=30)
days_attended = st.number_input("Days attended so far", min_value=0, value=25)
required_attendance = st.number_input("Required attendance (%)", min_value=50, max_value=100, value=75)

if st.button("Analyze Attendance"):

    # ---------- BASIC CALCULATIONS ----------
    total_classes = classes_per_day * total_working_days
    classes_done = classes_per_day * days_completed
    attended_classes = classes_per_day * days_attended
    leaves_taken = days_completed - days_attended

    if classes_done == 0:
        st.error("No classes completed yet.")
    else:
        current_attendance = (attended_classes / classes_done) * 100

        st.subheader("📊 Current Status")
        st.write(
            f"**Current Attendance:** {current_attendance:.2f}% "
            f"{attendance_emoji(current_attendance, required_attendance)}"
        )

        # ---------- LEAVES ----------
        max_leave_days = total_working_days - math.ceil(
            (required_attendance / 100) * total_working_days
        )
        remaining_leaves = max_leave_days - leaves_taken

        # ---------- DAYS NEEDED TO REACH GOAL ----------
        future_days = total_working_days - days_completed
        needed_days = None

        for i in range(future_days + 1):
            future_attended = attended_classes + (i * classes_per_day)
            future_total = classes_done + (i * classes_per_day)
            future_percentage = (future_attended / future_total) * 100
            if future_percentage >= required_attendance:
                needed_days = i
                break

        st.subheader("📈 Guidance")

        if needed_days is not None:
            st.success(
                f"You need to attend **next {needed_days} days continuously** "
                f"to reach {required_attendance}%."
            )
        else:
            st.error("Even attending all remaining classes will not reach the required attendance.")

        if remaining_leaves >= 0:
            st.info(
                f"You can take **{remaining_leaves} more leave days** "
                f"and still maintain {required_attendance}%."
            )
        else:
            st.warning("You have already exceeded the safe leave limit.")

        # ---------- ML CURRENT PREDICTION ----------
        attendance_percentage = current_attendance
        leave_percentage = 100 - attendance_percentage
        current_ml_input = np.array([[attendance_percentage, leave_percentage, 70]])
        current_risk = model.predict(current_ml_input)[0]

        st.subheader("🤖 ML Prediction")
        st.write(f"**Current Prediction:** {format_ml_risk(current_risk)}")

        # ---------- ML AFTER ATTENDING X DAYS ----------
        if needed_days is not None and needed_days > 0:
            future_attended_classes = attended_classes + (needed_days * classes_per_day)
            future_total_classes = classes_done + (needed_days * classes_per_day)
            future_attendance_percentage = (
                future_attended_classes / future_total_classes
            ) * 100

            future_ml_input = np.array(
                [[future_attendance_percentage, 100 - future_attendance_percentage, 70]]
            )
            future_risk = model.predict(future_ml_input)[0]

            st.write(
                f"**If you attend {needed_days} days → Attendance: "
                f"{future_attendance_percentage:.2f}% "
                f"{attendance_emoji(future_attendance_percentage, required_attendance)} "
                f"→ ML: {format_ml_risk(future_risk)}**"
            )

        # ---------- ML FOR Y+1 TO Y+7 LEAVES (FULL SEMESTER) ----------
if remaining_leaves >= 0:
    st.subheader("🧪 What if you take more leave days?")

    for extra in range(1, 8):
        total_leave_days = remaining_leaves + extra

        if total_leave_days <= total_working_days:

            # Final attended days assuming student attends all other days
            final_attended_days = (
                days_attended + (total_working_days - days_completed - extra)
            )

            if final_attended_days < 0:
                final_attended_days = 0

            final_attendance_percentage = (
                final_attended_days / total_working_days
            ) * 100

            # Emoji (goal-based)
            emoji_status = attendance_emoji(
                final_attendance_percentage, required_attendance
            )

            # ML prediction
            simulated_ml_input = np.array(
                [[final_attendance_percentage, 100 - final_attendance_percentage, 70]]
            )
            simulated_risk = model.predict(simulated_ml_input)[0]

            st.write(
                f"If you take **{total_leave_days} leave days** → "
                f"Final Attendance: **{final_attendance_percentage:.2f}%** "
                f"{emoji_status} → "
                f"ML: {format_ml_risk(simulated_risk)}"
            )
