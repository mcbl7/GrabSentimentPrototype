import streamlit as st
import pandas as pd
import joblib
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Grab Sentiment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #F7F9F8;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}


/* HERO */

.hero {
    background: white;
    border: 1px solid #E8ECEA;
    border-radius: 24px;
    padding: 42px;
    margin-bottom: 30px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.04);
}

.hero-badge {
    display: inline-block;
    background: #E8F8EF;
    color: #00A651;
    padding: 12px 18px;
    border-radius: 999px;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 18px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #657068;
    font-size: 17px;
    line-height: 1.65;
    max-width: 900px;
}


/* SECTION HEADERS */

.section-label {
    font-size: 13px;
    font-weight: 800;
    color: #00A651;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 5px;
}

.section-title {
    font-size: 26px;
    font-weight: 800;
    color: #17211B;
    margin-bottom: 4px;
}

.section-description {
    color: #748078;
    font-size: 14px;
    margin-bottom: 20px;
}


/* METRIC CARDS */

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E6ECE8;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.035);
}

div[data-testid="stMetricLabel"] {
    color: #7A857E;
    font-size: 13px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #17211B;
    font-size: 30px;
    font-weight: 800;
}


/* CONTAINERS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
}

[data-testid="stDataFrame"] {
    background: white;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #E7ECE9;
}


/* INPUTS */

textarea {
    border-radius: 14px !important;
}

div[data-baseweb="select"] > div {
    border-radius: 14px !important;
}


/* BUTTONS */

.stButton > button {
    background-color: #00B14F;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.15rem;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #009C45;
    color: white;
    border: none;
    transform: translateY(-1px);
}

.stButton > button:focus {
    box-shadow: none;
}


/* ALERTS */

div[data-testid="stAlert"] {
    border-radius: 14px;
}


/* DIVIDER */

hr {
    border: none;
    border-top: 1px solid #E8ECEA;
    margin: 32px 0;
}


/* CHARTS */

div[data-testid="stVegaLiteChart"] {
    background: white;
    border-radius: 18px;
    border: 1px solid #E8ECEA;
    padding: 12px;
}


/* PILLS */

.status-pill {
    display: inline-block;
    background: #EAF8EF;
    color: #008F45;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 700;
}

.warning-pill {
    display: inline-block;
    background: #FFF4E5;
    color: #A05A00;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 700;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #8A948E;
    font-size: 12px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid #E8ECEA;
}

</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        "data/reviews_cleaned.xlsx"
    )

    df["Store"] = (
        df["Store"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if "Review Date" in df.columns:

        df["Review Date"] = pd.to_datetime(
            df["Review Date"],
            errors="coerce",
        )

    return df


df = load_data()


# ============================================================
# RATING CLASS MAPPING
# ============================================================

def get_rating_class(sentiment):

    mapping = {
        "Negative": "1–2 stars",
        "Neutral": "3 stars",
        "Positive": "4–5 stars",
    }

    return mapping.get(
        sentiment,
        "Unknown",
    )


# ============================================================
# INITIALIZE REVIEW BROWSER
# ============================================================

if "demo_reviews" not in st.session_state:

    android_init = (
        df[
            df["Store"] == "android"
        ]
        .sort_values(
            "Review Date",
            ascending=False,
            na_position="last",
        )
        .head(5)
    )

    ios_init = (
        df[
            df["Store"] == "ios"
        ]
        .head(5)
    )

    seed_df = pd.concat(
        [
            android_init,
            ios_init,
        ],
        ignore_index=True,
    )

    seed_df["Rating Class"] = (
        seed_df["Sentiment"]
        .apply(get_rating_class)
    )

    st.session_state.demo_reviews = seed_df[
        [
            "Store",
            "Sentiment",
            "Rating Class",
            "Review Date",
            "Review Text",
        ]
    ].copy()


# ============================================================
# LOAD FINAL SELECTED MODEL
# ============================================================

@st.cache_resource
def load_prediction_pipeline():

    model = joblib.load(
        "models/final_cleaned/"
        "tfidf_svm_selected.joblib"
    )

    return model


model_pipeline = load_prediction_pipeline()


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">Grab Sentiment Analytics</div>'
    '<div class="hero-subtitle">'
    'Predicting customer satisfaction among Filipino Grab users '
    'using sentiment analysis, TF-IDF text representation, and a '
    'Linear Support Vector Machine classifier.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-label">Dataset</div>'
    '<div class="section-title">Dataset Overview</div>'
    '<div class="section-description">'
    'Summary of the cleaned review dataset used for '
    'model development and evaluation.'
    '</div>',
    unsafe_allow_html=True,
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Cleaned Reviews",
        f"{len(df):,}",
    )

with col2:
    st.metric(
        "Average Rating",
        f"{df['Star Rating'].mean():.2f} ⭐",
    )

with col3:
    st.metric(
        "Sentiment Classes",
        f"{df['Sentiment'].nunique()}",
    )


st.write("")


# ============================================================
# PLATFORM + SENTIMENT DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Platform & Sentiment Distribution'
    '</div>'
    '<div class="section-description">'
    'Distribution of cleaned reviews across platforms '
    'and sentiment target classes.'
    '</div>',
    unsafe_allow_html=True,
)


col_left, col_right = st.columns(
    2,
    gap="large",
)


with col_left:

    with st.container(border=True):

        st.markdown(
            "#### Reviews by Platform"
        )

        store_counts = (
            df["Store"]
            .value_counts()
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Android",
            f"{store_counts.get('android', 0):,}",
        )

        c2.metric(
            "iOS",
            f"{store_counts.get('ios', 0):,}",
        )

        platform_chart = pd.DataFrame(
            {
                "Platform": [
                    "Android",
                    "iOS",
                ],
                "Reviews": [
                    store_counts.get(
                        "android",
                        0,
                    ),
                    store_counts.get(
                        "ios",
                        0,
                    ),
                ],
            }
        )

        st.bar_chart(
            platform_chart,
            x="Platform",
            y="Reviews",
            use_container_width=True,
        )


with col_right:

    with st.container(border=True):

        st.markdown(
            "#### Sentiment Distribution"
        )

        sentiment_counts = (
            df["Sentiment"]
            .value_counts()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Positive",
            f"{sentiment_counts.get('Positive', 0):,}",
        )

        c2.metric(
            "Neutral",
            f"{sentiment_counts.get('Neutral', 0):,}",
        )

        c3.metric(
            "Negative",
            f"{sentiment_counts.get('Negative', 0):,}",
        )

        sentiment_chart = pd.DataFrame(
            {
                "Sentiment": [
                    "Negative",
                    "Neutral",
                    "Positive",
                ],
                "Reviews": [
                    sentiment_counts.get(
                        "Negative",
                        0,
                    ),
                    sentiment_counts.get(
                        "Neutral",
                        0,
                    ),
                    sentiment_counts.get(
                        "Positive",
                        0,
                    ),
                ],
            }
        )

        st.bar_chart(
            sentiment_chart,
            x="Sentiment",
            y="Reviews",
            use_container_width=True,
        )


st.divider()


# ============================================================
# NINE-MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-label">Model Selection</div>'
    '<div class="section-title">Nine-Model Comparison</div>'
    '<div class="section-description">'
    'Comparison of three text representations and three '
    'machine-learning classifiers using the cleaned dataset.'
    '</div>',
    unsafe_allow_html=True,
)


with st.container(border=True):

    st.markdown(
        '<span class="status-pill">'
        'Primary selection criterion · 5-fold CV Macro-F1'
        '</span>',
        unsafe_allow_html=True,
    )

    st.write("")

    model_comparison = pd.DataFrame(
        {
            "Representation": [
                "TF-IDF",
                "TF-IDF",
                "TF-IDF",
                "Word2Vec",
                "Word2Vec",
                "Word2Vec",
                "mBERT",
                "mBERT",
                "mBERT",
            ],
            "Classifier": [
                "SVM",
                "Naive Bayes",
                "Random Forest",
                "SVM",
                "Random Forest",
                "Gaussian Naive Bayes",
                "SVM",
                "Random Forest",
                "Gaussian Naive Bayes",
            ],
            "CV Macro-F1": [
                0.625449,
                0.577346,
                0.545196,
                0.591519,
                0.571184,
                0.500184,
                0.603459,
                0.531422,
                0.519157,
            ],
            "Test Accuracy": [
                0.828262,
                0.832109,
                0.792235,
                0.689402,
                0.805526,
                0.625044,
                0.763204,
                0.776146,
                0.622595,
            ],
            "Test Macro-F1": [
                0.627052,
                0.578244,
                0.551741,
                0.584013,
                0.578699,
                0.507339,
                0.579880,
                0.531686,
                0.515780,
            ],
        }
    )

    model_comparison = (
        model_comparison
        .sort_values(
            "CV Macro-F1",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "CV Macro-F1":
                st.column_config.NumberColumn(
                    format="%.4f"
                ),
            "Test Accuracy":
                st.column_config.NumberColumn(
                    format="%.4f"
                ),
            "Test Macro-F1":
                st.column_config.NumberColumn(
                    format="%.4f"
                ),
        },
    )

    st.write("")

    comparison_chart = (
        model_comparison[
            [
                "Representation",
                "Classifier",
                "CV Macro-F1",
            ]
        ]
        .copy()
    )

    comparison_chart["Model"] = (
        comparison_chart["Representation"]
        + " + "
        + comparison_chart["Classifier"]
    )

    comparison_chart = (
        comparison_chart[
            [
                "Model",
                "CV Macro-F1",
            ]
        ]
        .set_index(
            "Model"
        )
    )

    st.bar_chart(
        comparison_chart,
        use_container_width=True,
    )

    st.success(
        "**Selected model: TF-IDF + Linear SVM.** "
        "It achieved the highest 5-fold cross-validation "
        "Macro-F1 among the nine evaluated configurations."
    )

    st.caption(
        "Although TF-IDF + Naive Bayes achieved slightly "
        "higher test accuracy, model selection was based on "
        "cross-validation Macro-F1 to give equal importance "
        "to all three sentiment classes."
    )


st.divider()


# ============================================================
# FINAL MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-label">Evaluation</div>'
    '<div class="section-title">Final Model Performance</div>'
    '<div class="section-description">'
    'Performance of the selected TF-IDF + Linear SVM classifier '
    'on the cleaned held-out test set.'
    '</div>',
    unsafe_allow_html=True,
)


with st.container(border=True):

    perf1, perf2, perf3, perf4 = st.columns(4)

    perf1.metric(
        "Test Accuracy",
        "82.83%",
    )

    perf2.metric(
        "Test Macro-F1",
        "0.6271",
    )

    perf3.metric(
        "Weighted F1",
        "0.8232",
    )

    perf4.metric(
        "CV Macro-F1",
        "0.6254",
    )

    st.caption(
        "Best Linear SVM hyperparameter: C = 0.1"
    )


st.write("")


# ============================================================
# CLASS PERFORMANCE
# ============================================================

with st.container(border=True):

    st.markdown(
        "#### Performance by Sentiment Class"
    )

    class_performance = pd.DataFrame(
        {
            "Sentiment": [
                "Negative",
                "Neutral",
                "Positive",
            ],
            "Precision": [
                0.79,
                0.20,
                0.91,
            ],
            "Recall": [
                0.90,
                0.14,
                0.84,
            ],
            "F1 Score": [
                0.84,
                0.16,
                0.87,
            ],
            "Test Support": [
                1229,
                153,
                1477,
            ],
        }
    )

    st.dataframe(
        class_performance,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<span class="warning-pill">'
        'Neutral class limitation'
        '</span>',
        unsafe_allow_html=True,
    )

    st.write("")

    st.info(
        "Neutral reviews represented the smallest class "
        "and were the most difficult sentiment category "
        "for the final classifier."
    )


st.divider()


# ============================================================
# PLATFORM PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-label">Platform Evaluation</div>'
    '<div class="section-title">Android vs iOS Performance</div>'
    '<div class="section-description">'
    'Performance of the final classifier on platform-specific '
    'subsets of the cleaned held-out test set.'
    '</div>',
    unsafe_allow_html=True,
)


with st.container(border=True):

    metric_left, metric_right = st.columns(
        2,
        gap="large",
    )


    with metric_left:

        st.markdown(
            "#### Android"
        )

        a1, a2 = st.columns(2)

        a1.metric(
            "Accuracy",
            "89.76%",
        )

        a2.metric(
            "Macro-F1",
            "0.6542",
        )

        st.caption(
            "859 reviews in the cleaned test set."
        )


    with metric_right:

        st.markdown(
            "#### iOS"
        )

        i1, i2 = st.columns(2)

        i1.metric(
            "Accuracy",
            "79.85%",
        )

        i2.metric(
            "Macro-F1",
            "0.6125",
        )

        st.caption(
            "2,000 reviews in the cleaned test set."
        )


    platform_metrics_chart = pd.DataFrame(
        {
            "Platform": [
                "Android",
                "iOS",
            ],
            "Accuracy": [
                0.897555,
                0.798500,
            ],
            "Macro F1": [
                0.654218,
                0.612487,
            ],
        }
    ).set_index(
        "Platform"
    )

    st.bar_chart(
        platform_metrics_chart,
        use_container_width=True,
    )

    st.caption(
        "These values describe observed test-set performance. "
        "No causal explanation for the platform difference "
        "is assumed."
    )


st.divider()


# ============================================================
# CUSTOMER DISSATISFACTION FACTORS
# ============================================================

st.markdown(
    '<div class="section-label">Negative Review Analysis</div>'
    '<div class="section-title">'
    'Customer Dissatisfaction Factors'
    '</div>'
    '<div class="section-description">'
    'Operational complaint categories identified through '
    'keyword-in-context analysis of reviews predicted as Negative.'
    '</div>',
    unsafe_allow_html=True,
)


with st.container(border=True):

    negative1, negative2 = st.columns(2)

    negative1.metric(
        "Predicted Negative Reviews",
        "6,750",
    )

    negative2.metric(
        "Analysis Method",
        "KWIC",
    )

    st.caption(
        "A negative review may match more than one factor, "
        "so percentages are not expected to total 100%."
    )

    dissatisfaction_df = pd.DataFrame(
        {
            "Factor": [
                "Application Performance",
                "Order Fulfillment & Accuracy",
                "Driver/Rider/Courier Conduct",
                "Payment & Refund",
                "Customer Support",
                "Waiting Time & Delay",
                "Pricing & Fare",
            ],
            "Matched Reviews": [
                2438,
                2332,
                2014,
                1528,
                1166,
                979,
                933,
            ],
            "Percentage": [
                36.12,
                34.55,
                29.84,
                22.64,
                17.27,
                14.50,
                13.82,
            ],
        }
    )

    st.write("")

    st.dataframe(
        dissatisfaction_df,
        use_container_width=True,
        hide_index=True,
    )

    kwic_chart = (
        dissatisfaction_df[
            [
                "Factor",
                "Percentage",
            ]
        ]
        .set_index(
            "Factor"
        )
    )

    st.bar_chart(
        kwic_chart,
        use_container_width=True,
    )


st.divider()


# ============================================================
# LDA TOPIC MODELING
# ============================================================

st.markdown(
    '<div class="section-label">Topic Modeling</div>'
    '<div class="section-title">'
    'Negative Review Topics'
    '</div>'
    '<div class="section-description">'
    'Six dominant topics identified through Latent Dirichlet '
    'Allocation after Filipino and English stopword filtering.'
    '</div>',
    unsafe_allow_html=True,
)


with st.container(border=True):

    lda_topics = pd.DataFrame(
        {
            "Topic": [
                "Topic 1",
                "Topic 2",
                "Topic 3",
                "Topic 4",
                "Topic 5",
                "Topic 6",
            ],
            "Interpreted Theme": [
                "Food delivery, riders, and order issues",
                "Ride booking, fares, and pricing",
                "Cancellations, waiting time, and delays",
                "Payments, refunds, cards, and wallets",
                "Location, address, and application issues",
                "Customer support and service assistance",
            ],
            "Reviews": [
                1337,
                954,
                1459,
                1004,
                1022,
                974,
            ],
            "Percentage": [
                19.81,
                14.13,
                21.61,
                14.87,
                15.14,
                14.43,
            ],
        }
    )

    st.dataframe(
        lda_topics,
        use_container_width=True,
        hide_index=True,
    )

    topic_chart = (
        lda_topics[
            [
                "Interpreted Theme",
                "Percentage",
            ]
        ]
        .set_index(
            "Interpreted Theme"
        )
    )

    st.bar_chart(
        topic_chart,
        use_container_width=True,
    )

    st.caption(
        "Each negative review was assigned to its dominant "
        "LDA topic, so the percentages total approximately 100%."
    )

    with st.expander(
        "View representative topic keywords"
    ):

        st.markdown(
            """
**Topic 1 — Food delivery, riders, and order issues**  
food, delivery, drivers, order, rider, driver, riders, cancel, orders, time

**Topic 2 — Ride booking, fares, and pricing**  
drivers, book, ride, uber, booking, price, fare, expensive, high, taxi

**Topic 3 — Cancellations, waiting time, and delays**  
order, driver, time, cancel, cancelled, hour, waiting, wait, booking, minutes

**Topic 4 — Payments, refunds, cards, and wallets**  
payment, money, account, card, cash, pay, refund, charged, wallet, gcash

**Topic 5 — Location, address, and application issues**  
update, fix, location, address, phone, number, pin, keeps, wrong, work

**Topic 6 — Customer support and service assistance**  
service, customer, support, help, issue, chat, poor, worst, contact, agent
            """
        )


st.divider()


# ============================================================
# REAL-TIME SENTIMENT PREDICTION
# ============================================================

st.markdown(
    '<div class="section-label">Final Model</div>'
    '<div class="section-title">'
    'Real-Time Sentiment Prediction'
    '</div>'
    '<div class="section-description">'
    'Test the final TF-IDF + Linear SVM classifier '
    'using a new Grab review.'
    '</div>',
    unsafe_allow_html=True,
)


with st.container(border=True):

    st.markdown(
        '<span class="status-pill">'
        'Final selected model · TF-IDF + Linear SVM'
        '</span>',
        unsafe_allow_html=True,
    )

    st.write("")

    col_input1, col_input2 = st.columns(
        [3, 1],
        gap="large",
    )


    with col_input1:

        user_input = st.text_area(
            "Enter Grab user review:",
            placeholder=(
                "Example: The rider was polite "
                "but my order arrived very late."
            ),
            height=130,
        )


    with col_input2:

        input_platform = st.selectbox(
            "Platform Source:",
            [
                "android",
                "ios",
            ],
            format_func=lambda x: (
                "Android"
                if x == "android"
                else "iOS"
            ),
        )


    predict_clicked = st.button(
        "Predict Sentiment",
        type="primary",
    )


    if predict_clicked:

        if user_input.strip() != "":

            review_text = (
                user_input.strip()
            )

            pred_label = (
                model_pipeline.predict(
                    [review_text]
                )[0]
            )

            rating_class = (
                get_rating_class(
                    pred_label
                )
            )

            clf = (
                model_pipeline
                .named_steps["clf"]
            )

            decision_scores = (
                model_pipeline
                .decision_function(
                    [review_text]
                )[0]
            )

            classes = list(
                clf.classes_
            )

            score_dict = dict(
                zip(
                    classes,
                    decision_scores,
                )
            )

            neg_score = score_dict.get(
                "Negative",
                0.0,
            )

            neu_score = score_dict.get(
                "Neutral",
                0.0,
            )

            pos_score = score_dict.get(
                "Positive",
                0.0,
            )


            if pred_label == "Positive":

                st.success(
                    "### 😊 Positive\n"
                    f"Associated rating class: "
                    f"**{rating_class}**"
                )


            elif pred_label == "Neutral":

                st.warning(
                    "### 😐 Neutral\n"
                    f"Associated rating class: "
                    f"**{rating_class}**"
                )


            else:

                st.error(
                    "### 😡 Negative\n"
                    f"Associated rating class: "
                    f"**{rating_class}**"
                )


            with st.expander(
                "View classifier decision margins"
            ):

                st.caption(
                    "SVM decision margins are relative "
                    "classifier scores, not probabilities."
                )

                margin_cols = st.columns(3)

                margin_cols[0].metric(
                    "Negative",
                    f"{neg_score:.3f}",
                )

                margin_cols[1].metric(
                    "Neutral",
                    f"{neu_score:.3f}",
                )

                margin_cols[2].metric(
                    "Positive",
                    f"{pos_score:.3f}",
                )


            new_row = pd.DataFrame(
                [
                    {
                        "Store":
                            input_platform,

                        "Sentiment":
                            pred_label,

                        "Rating Class":
                            rating_class,

                        "Review Date":
                            datetime.now(),

                        "Review Text":
                            review_text,
                    }
                ]
            )


            st.session_state.demo_reviews = (
                pd.concat(
                    [
                        new_row,
                        st.session_state.demo_reviews,
                    ],
                    ignore_index=True,
                )
            )


            st.toast(
                "Review classified and added "
                "to the Review Browser.",
                icon="✅",
            )


        else:

            st.info(
                "Please enter review text "
                "before predicting."
            )


st.divider()


# ============================================================
# REVIEW BROWSER
# ============================================================

st.markdown(
    '<div class="section-label">Review Records</div>'
    '<div class="section-title">Review Browser</div>'
    '<div class="section-description">'
    'Displays selected historical reviews and live sentiment '
    'predictions from the current session. Rating Class reflects '
    'the sentiment-label mapping used in the study.'
    '</div>',
    unsafe_allow_html=True,
)


with st.container(border=True):

    st.dataframe(
        st.session_state.demo_reviews,
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    if st.button(
        "Reset Review Browser"
    ):

        del st.session_state.demo_reviews

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Grab Sentiment Analytics · BSIT Data Analytics Capstone'
    '<br>'
    'Final selected model: TF-IDF + Linear SVM'
    '</div>',
    unsafe_allow_html=True,
)