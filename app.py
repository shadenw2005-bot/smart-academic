import streamlit as st
import pandas as pd
import re
import string
import time

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Smart Academic Assistant",
    page_icon="🎓",
    layout="wide"
)

# =========================
# Helper Functions
# =========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", " ", text)  # remove digits
    text = re.sub(f"[{string.punctuation}]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text


def build_dataset() -> pd.DataFrame:
    text_data = {
        "text": [
            # ---------------- Task ----------------
            "Finish math assignment",
            "Complete programming project",
            "Write research paper",
            "Prepare presentation for class",
            "Submit lab report",
            "Read chapter 5",
            "Solve physics problems",
            "Work on group project",
            "Complete online quiz",
            "Review lecture notes",
            "Do my homework",
            "Finish homework",
            "Complete assignment tonight",
            "Study for class",
            "Do my project work",
            "Finish AI assignment",
            "Submit project",
            "Work on final report",
            "Prepare class notes",
            "Do the lab task",
            "Complete the worksheet",
            "Work on my coding task",
            "Start my university assignment",
            "Prepare the report draft",
            "I need to finish my homework",
            "I need to do my coursework",
            "Work on the submission",
            "Edit my project report",
            "Continue my assignment",
            "Complete the class task",

            # ---------------- Exam ----------------
            "Final exam tomorrow",
            "Midterm exam next week",
            "Calculus exam on Monday",
            "Physics test this Friday",
            "Chemistry final exam",
            "English exam preparation",
            "Programming exam practice",
            "Statistics test",
            "Biology exam review",
            "History final test",
            "Exam in 2 days",
            "Quiz tomorrow",
            "I have a test next week",
            "Study for the final exam",
            "Upcoming midterm exam",
            "My exam is very soon",
            "There is a quiz tomorrow",
            "I need to prepare for the test",
            "My final is next week",
            "The exam is after two days",
            "Exam coming up",
            "Prepare for midterm",
            "Test on Monday",
            "Quiz in three days",
            "Revision for exam",

            # ---------------- Reminder ----------------
            "Submit assignment by 5pm",
            "Meeting with advisor",
            "Register for next semester",
            "Pay tuition fee",
            "Buy textbooks",
            "Deadline for project proposal",
            "Attend workshop",
            "Library books due",
            "Scholarship application deadline",
            "Group meeting at 3pm",
            "Do not forget the deadline",
            "Remember to submit the task",
            "Meeting tomorrow morning",
            "Project due tonight",
            "Assignment deadline today",
            "Reminder to attend the class meeting",
            "Don't forget to register",
            "Submit before deadline",
            "Workshop tomorrow",
            "Pay fees today",
            "Library return reminder",
            "Advisor meeting today",
            "Deadline is tonight",
            "Meeting reminder",
            "Due date reminder",

            # ---------------- Other ----------------
            "Hello how are you",
            "What time is lunch",
            "Weather is nice today",
            "Call mom later",
            "Watch movie tonight",
            "Gym session at 6pm",
            "Buy groceries",
            "Weekend plans",
            "Check social media",
            "Listen to music",
            "Hangout with friends",
            "Go out with friends",
            "Meet my friends",
            "Chill with friends",
            "Spend time with friends",
            "Let's go shopping",
            "I want to relax today",
            "Watch a series tonight",
            "Go for coffee",
            "I am bored today",
            "Need to call my friend",
            "Plan a trip",
            "Play games tonight",
            "Take a walk outside",
            "Scroll through Instagram"
        ],
        "label": (
            ["Task"] * 30 +
            ["Exam"] * 25 +
            ["Reminder"] * 25 +
            ["Other"] * 25
        )
    }

    df = pd.DataFrame(text_data)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


@st.cache_resource
def train_model():
    df = build_dataset()

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [True, False],
        "clf__C": [0.5, 1.0, 2.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "best_params": grid.best_params_
    }

    return best_model, metrics, df


def predict_text(text: str, model) -> str:
    cleaned = clean_text(text)
    return model.predict([cleaned])[0]


def count_tasks(predictions):
    task_count = sum(1 for p in predictions if p == "Task")
    exam_count = sum(1 for p in predictions if p == "Exam")
    reminder_count = sum(1 for p in predictions if p == "Reminder")
    other_count = sum(1 for p in predictions if p == "Other")
    return task_count, exam_count, reminder_count, other_count


def calculate_stress(task_count: int, exam_count: int, days_to_exam: int) -> str:
    if exam_count > 0 and days_to_exam <= 3:
        return "High"
    if task_count >= 4:
        return "High"
    if task_count >= 2 or exam_count > 0:
        return "Medium"
    return "Low"


def stress_rules_engine(stress_level: str, days_to_exam: int, task_count: int):
    alerts = []
    suggestions = []

    if days_to_exam <= 3:
        alerts.append("⚠️ Your exam is very close.")
        suggestions.append("Start revising immediately and focus on high-priority topics.")

    if stress_level == "High":
        alerts.append("🚨 High stress detected.")
        suggestions.append("Take short breaks and reduce overload where possible.")
    elif stress_level == "Medium":
        alerts.append("⚠️ Moderate stress level.")
        suggestions.append("Manage your time carefully and organize your priorities.")

    if task_count >= 5:
        alerts.append("📚 You have too many tasks.")
        suggestions.append("Break tasks into smaller steps and prioritize urgent ones.")

    if not alerts:
        suggestions.append("✅ No major risk detected. Keep following your current study plan.")

    return alerts, suggestions


# =========================
# Train Once
# =========================
model, metrics, df = train_model()

# =========================
# UI Header
# =========================
st.title("🎓 Smart Academic Assistant")
st.markdown(
    """
This tool analyzes student academic inputs using **AI-based text classification**
and generates **stress-aware alerts and suggestions**.

**What it does:**
- Classifies each input as **Task / Exam / Reminder / Other**
- Estimates academic stress level
- Provides alerts and recommendations
"""
)

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Student Input")
    user_input = st.text_area(
        "Enter your tasks or exam-related text (one item per line):",
        height=180,
        placeholder="Finish AI assignment\nFinal exam tomorrow\nSubmit project"
    )
    days_to_exam = st.slider("Days to Exam", 0, 30, 5)
    analyze_clicked = st.button("Analyze", use_container_width=True)

with right_col:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    st.metric("Precision", f"{metrics['precision']:.2f}")
    st.metric("Recall", f"{metrics['recall']:.2f}")
    st.metric("F1 Score", f"{metrics['f1']:.2f}")

    with st.expander("Best Parameters"):
        st.json(metrics["best_params"])

    with st.expander("Dataset Information"):
        st.write(f"Total examples: **{len(df)}**")
        st.write("Classes: **Task, Exam, Reminder, Other**")

    with st.expander("Quick Example Inputs"):
        st.code(
            "Finish AI assignment\nFinal exam tomorrow\nSubmit project",
            language="text"
        )

if analyze_clicked:
    start_time = time.time()

    lines = [line.strip() for line in user_input.split("\n") if line.strip()]

    if not lines:
        st.warning("Please enter at least one valid input line.")
    else:
        predictions = [predict_text(text, model) for text in lines]

        task_count, exam_count, reminder_count, other_count = count_tasks(predictions)
        stress_level = calculate_stress(task_count, exam_count, days_to_exam)
        alerts, suggestions = stress_rules_engine(stress_level, days_to_exam, task_count)

        response_time = time.time() - start_time

        st.divider()

        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:
            st.subheader("📊 Predicted Labels")
            results_df = pd.DataFrame({
                "Input Text": lines,
                "Predicted Label": predictions
            })
            st.dataframe(results_df, use_container_width=True, hide_index=True)

        with result_col2:
            st.subheader("🧠 Stress Summary")
            st.write(f"**Task Count:** {task_count}")
            st.write(f"**Exam Count:** {exam_count}")
            st.write(f"**Reminder Count:** {reminder_count}")
            st.write(f"**Other Count:** {other_count}")
            st.write(f"**Stress Level:** {stress_level}")

        st.subheader("⚡ Performance")
        st.metric("Response Time (seconds)", f"{response_time:.3f}")

        st.subheader("🚨 Alerts")
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.info("No alerts generated.")

        st.subheader("💡 Suggestions")
        for suggestion in suggestions:
            st.success(suggestion)

        with st.expander("Detailed Classification Report"):
            st.text(metrics["report"])

st.divider()
st.caption("Built for AI Application Programming project - Smart Academic Reminder and Stress Monitor")
