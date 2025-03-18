# Transformer-Based Disease Progression Prediction

## Abstract

Accurate prediction of disease progression remains an ongoing challenge within healthcare informatics, particularly when clinicians attempt to leverage unstructured clinical notes alongside structured diagnostic codes. This project presents a transformer-based approach—ClinicalBERT—to predict patient disease progression by analyzing discharge summaries and structured diagnosis data from the comprehensive Medical Information Mart for Intensive Care (MIMIC-IV) dataset. Specifically, this study aims to predict the onset of a secondary condition (e.g., sepsis) based on prior medical encounters that document a related primary condition (e.g., pneumonia).

Clinical narratives in discharge summaries offer nuanced, context-rich information that traditional structured data alone often fail to capture. In contrast, structured International Classification of Diseases (ICD-9 and ICD-10) codes provide definitive, clinically validated diagnoses and timelines. This work integrates these two modalities—unstructured text and structured clinical coding—to develop a powerful predictive framework capable of identifying patients at risk of disease progression early. The methodology involves fine-tuning ClinicalBERT alongside conventional baseline models and evaluating performance through explainable AI methods such as SHAP analysis and transformer attention visualization.

The results show substantial improvements in predictive accuracy and model interpretability compared to traditional machine learning baselines, demonstrating the potential of transformer-based models in clinical risk stratification and decision support.

---

## Project Objectives

- **Leverage ClinicalBERT**—a domain-specific transformer model—to effectively capture context from clinical notes and accurately predict subsequent disease progression events.
- **Integrate structured diagnosis data (ICD-9 and ICD-10)** from MIMIC-IV on FHIR, enhancing labeling precision and reliability.
- **Address missing structured ICD codes** by dynamically retrieving standardized mappings via external medical classification APIs (WHO ICD API).
- **Benchmark ClinicalBERT performance** against classical baseline approaches, specifically TF-IDF with Logistic Regression.
- **Provide interpretability and clinical insights** through SHAP (SHapley Additive exPlanations) and attention mechanisms inherent to transformer models.

---

## Methodological Framework

### Data Extraction & Preprocessing
Clinical narratives from MIMIC-IV-Note were systematically preprocessed, including tokenization, normalization, stopword removal (with retention of key clinical terminology), and section segmentation to facilitate meaningful transformer input. ICD-9 and ICD-10 diagnoses were extracted from structured MIMIC-IV on FHIR data, providing verified labels of patient conditions across multiple admissions.

### Feature Engineering
Clinical notes were encoded into embedding vectors using ClinicalBERT. A baseline model, leveraging Term Frequency-Inverse Document Frequency (TF-IDF) vectorization, provided comparative insights into embedding effectiveness. Embeddings were cached to optimize computational resources, enabling iterative experimentation and model refinement.

### Model Training and Evaluation
ClinicalBERT was fine-tuned to perform binary classification tasks predicting disease progression (e.g., pneumonia to sepsis) using labeled embeddings. Training employed sophisticated methods including gradient accumulation, mixed-precision training, and early stopping. Models were rigorously evaluated with metrics such as Area Under Receiver Operating Characteristic Curve (AUROC), precision, recall, and F1-score, and compared to traditional baseline methods.

### Model Interpretability and Explainability
Explainability was prioritized, employing SHAP to elucidate critical textual predictors within clinical notes and transformer attention mechanisms to visually dissect model decision-making processes. These approaches offered clinicians clear, actionable insights into disease progression predictions.

---

## Significance and Contributions

This research advances the predictive capabilities of clinical decision support systems (CDSS) through the effective combination of structured diagnostic codes and unstructured clinical narratives using advanced transformer-based language models. The integration of multimodal data sources enhances model performance, allowing healthcare practitioners to identify high-risk patient groups proactively and facilitating timely intervention to improve clinical outcomes.

---

## Ethical Considerations and Data Usage Note

The data utilized in this research, sourced from the MIMIC-IV dataset, adhere strictly to PhysioNet’s established data use agreements to ensure patient confidentiality and privacy. Consequently, the dataset or derived patient-specific data are not included within this repository. Researchers interested in accessing these data must obtain explicit authorization from PhysioNet, complying with all stipulated conditions of use and patient privacy protections.

For detailed information and access protocols, please consult the [PhysioNet MIMIC-IV database](https://physionet.org/content/mimiciv/).

---

## References

The following works significantly influenced the development of methodologies and model selection employed in this project:

- Alsentzer, E., Murphy, J., Boag, W., Weng, W. H., Jin, D., Naumann, T., & McDermott, M. (2019). ClinicalBERT: Modeling clinical notes and predicting hospital readmission. *arXiv preprint arXiv:1904.05342.* https://arxiv.org/abs/1904.05342

- Rasmy, L., Xiang, Y., Xie, Z., Tao, C., & Zhi, D. (2021). Med-BERT: Pre-trained contextualized embeddings on large-scale structured electronic health records for disease prediction. *NPJ Digital Medicine, 4*(1), Article 86. https://doi.org/10.1038/s41746-021-00455-y

- Gao, J., Li, X., Wang, Y., Chen, C., & Xu, H. (2022). AD-BERT: Using pre-trained contextualized embeddings to predict the progression from mild cognitive impairment to Alzheimer's disease. *arXiv preprint arXiv:2203.00150.* https://arxiv.org/abs/2203.00150

- Yang, Z., Yu, X., Liu, L., Yang, L., Liu, Y., & Jiang, Y. (2023). Large language multimodal models for 5-year chronic disease cohort prediction using EHR data. *arXiv preprint arXiv:2305.12894.* https://arxiv.org/abs/2305.12894

- Chen, M. C., Ball, R. L., Yang, L., Moradzadeh, N., Chapman, B. E., Larson, D. B., & Langlotz, C. P. (2021). Deep learning to classify radiology free-text reports. *Radiology, 300*(3), 607–616. https://doi.org/10.1148/radiol.2021204252

- Pang, C., Jiang, X., & Kalluri, K. S. (2022). Transformer-based active learning for multi-class text annotation in healthcare. *Journal of Biomedical Informatics, 130*, Article 104060. https://doi.org/10.1016/j.jbi.2022.104060

- Choi, E., Bahadori, M. T., Searles, E., Coffey, C., & Sun, J. (2016). Multi-layer representation learning for medical concepts. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 1495–1504. https://doi.org/10.1145/2939672.2939823

---

## License
This project is released under the MIT License.
Note: Access to MIMIC-IV data requires PhysioNet approval, and any derived datasets must adhere to PhysioNet's data-sharing policies.