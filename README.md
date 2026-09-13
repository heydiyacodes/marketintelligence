# Competitive Intelligence & Customer Churn Analytics

## Project Overview

This project builds a framework to predict customer churn using external competitive market signals as features. The core hypothesis: **customers in categories facing competitive threat have higher churn propensity**.

The project combines:
1. **Competitor risk scoring** - Analyzing 50+ beauty brands across market signals
2. **Customer churn prediction** - ML models linking competitor threat to customer behavior
3. **Business impact quantification** - Identifying vulnerable customer segments and ARR at risk

---

## Business Problem

Most companies track competitive moves and customer retention in silos. This creates a blind spot: **external competitive threats directly impact customer churn, but are rarely quantified in churn models.**

### Key Questions We Answer
- Do customers in categories with expanding competitors churn more?
- Which customer segments are most vulnerable to competitive threat?
- How much revenue is at risk from high-threat competitors?
- Can we predict churn better by including competitive signal features?

---

## Approach

### Phase 1: Competitor Risk Scoring

**Data Source:** 50 Indian beauty brands with known market outcomes
- **Thriving:** Nykaa, Minimalist, Plum, The Derma Co (expanding into new categories)
- **Acquired:** Dot & Key (Unilever 2020), Mama Earth (Unilever 2021)
- **Stalled:** Mamas Organic (no funding since 2022)
- **Stable/Declining:** Global brands with limited expansion

**Signals Engineered:**
- Hiring velocity (from LinkedIn job postings)
- Product launch frequency (from company websites/news)
- Sentiment analysis (from news mentions and reviews)
- Pricing changes (from marketplace data)
- Funding/acquisition news

**Expansion Threat Score:** Brands ranked 0-100 based on likelihood to expand into new categories
- High threat (75-100): Actively hiring, launching products, positive sentiment
- Medium threat (35-74): Stable presence, selective growth
- Low threat (0-34): Declining, stalled, or acquired

### Phase 2: Customer Segmentation

**Data Source:** Real telecom customer churn dataset (7,043 customers, 26% churn rate)

**Category Mapping:** Telecom services → Beauty product categories
- Internet Service → Primary Category (Skincare, Makeup, Haircare, Accessories)
- Engagement level → Multicategory buyer (yes/no)
- Spend → Monthly investment in category

**Customer Segments:** By monthly spend
- Budget: $0-30/month (1,653 customers)
- Mid-tier: $30-60/month (1,265 customers)
- Premium: $60-100/month (3,223 customers)
- Enterprise: $100+/month (902 customers)

### Phase 3: Feature Engineering

For each customer, we calculated:
1. **Category-level threat score** - Average expansion threat of competitors in their primary category
2. **Individual threat variance** - Added realistic variance (±8 points) so customers in same category face different threat levels
3. **Threat tier assignment** - Low (0-33), Medium (34-66), High (67-100)

**Why variance matters:** Customers in same category may prefer different brands. Some buy from high-growth players (high threat), others from established brands (low threat).

### Phase 4: Statistical Validation

**Chi-Square Test:**
- H0: Competitor threat tier is independent of churn
- Result: χ² = 95.32, p < 0.001
- **Conclusion:** Competitor threat SIGNIFICANTLY predicts churn (reject H0)

**Churn Rates by Threat Tier:**
- Low threat: 28.4%
- Medium threat: 30.2%
- High threat: 35.1%
- **Variance:** 1.24x higher churn in high-threat tier

### Phase 5: Predictive Modeling

**Model 1: Logistic Regression**
- Features: competitor_threat_score, tenure, monthly_charges, category_engagement
- Train AUC: 0.76
- Test AUC: 0.74
- Cross-validation (k=5): 0.73 ± 0.04

**Model 2: Random Forest**
- Features: Same as above
- Train AUC: 0.84
- Test AUC: 0.81
- Cross-validation (k=5): 0.79 ± 0.05

**Feature Importance (Random Forest):**
1. Competitor threat score: 28%
2. Tenure: 24%
3. Monthly charges: 22%
4. Category engagement: 18%
5. Segment: 8%

**Key Finding:** Competitor threat is the 2nd strongest predictor after tenure, indicating external competitive moves matter for churn.

### Phase 6: Segment-Level Impact Analysis

**Churn by Segment × Threat Tier:**

| Segment | Low Threat | Medium Threat | High Threat | Variance |
|---------|-----------|---------------|------------|----------|
| Budget | 30.8% | 10.3% | 8.1% | 0.26x |
| Mid-tier | 25.8% | 25.9% | 33.3% | 1.29x |
| Premium | 29.6% | 35.7% | 40.0% | 1.35x |
| Enterprise | 29.7% | 27.7% | 33.3% | 1.12x |

**Most Vulnerable:** Premium segment shows 1.35x churn variance between low and high-threat competitors.

---

## Technical Stack

**Data Processing & Analysis:**
- Python 3.x
- Pandas, NumPy
- PostgreSQL (Supabase cloud)

**ML & Statistics:**
- Scikit-learn (Logistic Regression, Random Forest)
- SciPy (Chi-square, ANOVA, cross-validation)

**Visualization:**
- Matplotlib, Seaborn (Python)
- Power BI (dashboard, not included in repo)

---

## How to Use This Project

### 1. Understand the Framework
```python
# Load data
import pandas as pd
churn_data = pd.read_csv("telco_churn.csv")

# Assign competitor threat (example)
churn_data['competitor_threat_score'] = [score for each customer based on their category]

# Predict churn
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
churn_predictions = model.predict_proba(X_test)
```

### 2. Validate on Your Data
- Replace the 7K telecom customers with your actual customer base
- Map your product categories to competitor threat scores (use provided competitor_threat_df)
- Run chi-square test to validate signal in your data
- Retrain models on your churn labels

### 3. Segment Analysis
- Identify which customer segments are most vulnerable to competitor threat
- Quantify ARR at risk per segment
- Prioritize retention campaigns in high-vulnerability segments

---

## Files in This Repo
