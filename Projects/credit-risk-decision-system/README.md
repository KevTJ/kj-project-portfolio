PROJECT RESULTS

Data
- Training observations:
- Test observations:
- Raw features:
- Default rate:

Modeling
- Logistic ROC-AUC:
- Random Forest ROC-AUC:
- XGBoost ROC-AUC:
- XGBoost PR-AUC:
- Brier score:

Business
- Optimal approval threshold:
- Approval rate:
- Portfolio default rate:
- Expected profit:
- Loss reduction vs baseline:

Engineering
- Final feature count:
- API latency:
- Test coverage:


Notebook 1 — Design decisions: 
Formulated credit risk as a probability-estimation problem embedded within a lending decision system; separated predictive performance from lending policy; established asymmetric costs of false approvals and false rejections; introduced expected-value-based thresholding and portfolio-level approval-risk tradeoffs; identified calibration, leakage prevention, and business interpretation as core requirements.