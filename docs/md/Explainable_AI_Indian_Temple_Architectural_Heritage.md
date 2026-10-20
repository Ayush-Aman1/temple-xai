          Explainable Multimodal AI for Recognition and
           Interpretation of Indian Temple Architectural
                               Heritage
                                           Complete Research Project Documentation


           Indian Knowledge Systems (IKS) — Unit IV: Town Planning & Architecture / Temple
                                           Architecture


                         Proposed project for an undergraduate CSE research-oriented project

     Project focus: Computer Vision + Deep Learning + Explainable AI + Indian Architectural Knowledge + Digital
     Heritage

     Recommended research direction: A feature-aware system that identifies temple architectural traditions,
     detects meaningful architectural components, and provides evidence-grounded explanations instead of relying
     only on whole-image classification.




IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                     Page 1
    1. Executive Summary
    Indian temple architecture represents a highly structured body of architectural knowledge involving spatial organization,
    geometry, construction, ornamentation and regional traditions. This project proposes an Explainable Multimodal
    Artificial Intelligence framework for automated recognition and interpretation of Indian temple architectural heritage.
    Rather than producing only a style label, the system will identify architecturally meaningful visual features such as
    shikhara/vimana forms, gopurams, mandapas, pillars, bases and ornamental structures, and use these features to
    support classification.

    The central research idea is to combine whole-image visual representations with architectural component information
    and a structured IKS knowledge layer. Explainable AI methods such as Grad-CAM and feature attribution will show
    which regions and architectural components influenced a prediction. A temple-disjoint evaluation protocol will be used
    so that photographs of the same temple do not appear in both training and test sets.

     Temple Image
     ↓
     Image Preprocessing
     ↓
     Architectural Component Detection
     ↓
     Feature Extraction
     ↓
     Feature Fusion
     ↓
     Style Classification
     ↓
     Explainable AI
     ↓
     IKS Knowledge Mapping
     ↓
     Human-readable Architectural Explanation



    2. Proposed Research Title
    Primary title: Explainable Multimodal AI for Recognition and Interpretation of Indian Temple Architectural Heritage

    Alternative title: A Feature-Aware and Explainable Deep Learning Framework for Automated Classification of Indian
    Temple Architectural Heritage


    3. IKS Syllabus Alignment
      Syllabus component                           Project connection

      Unit IV — Indian Architecture                Core domain: temple architecture and architectural traditions.

      V■stu ■■stra                                 Used as part of the structured architectural knowledge layer where appropriate.

      Temple Architecture                          Primary knowledge domain and annotation taxonomy.

      Town Planning & Architecture                 Contextual analysis of spatial and structural features.

      Unit II — Linguistics / Knowledge FrameworksOptional textual knowledge layer and provenance-aware representation.

      Unit III — Science, Engineering & Technology Computer vision, computational analysis and digital preservation.




    4. Why This Topic Is Important
•     Indian temples contain diverse structural, geometric and decorative features that can be digitally documented and
      analyzed.

•     Manual architectural identification requires domain expertise and can be time-consuming.




IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                                       Page 2
•     Computer vision can support heritage documentation, education, digital archives and virtual heritage applications.

•     Explainability is important because a heritage-analysis system should show architectural evidence for a prediction
      rather than only output a class label.

•     The project provides a direct bridge between an IKS knowledge domain and modern CSE research methods.



    5. Existing Research and the Need for a New Contribution
    Recent research has already demonstrated deep-learning classification of Indian temple architectural styles. A 2026
    study reported a dataset of 9,046 temple images from 218 temples and compared EfficientNet-B0, MobileNetV2,
    MobileNetV3, GhostNet and ShuffleNetV2 for N■gara, Vesara and Dr■vi■a classification. The study also used
    Grad-CAM for interpretation.

    Therefore, a project framed simply as “CNN-based classification of Indian temple architecture” would largely reproduce
    an existing research direction. The proposed work instead focuses on architectural component detection,
    feature-aware classification, explanation alignment and knowledge-grounded interpretation.

       Proposed research gap: Existing work demonstrates that deep-learning models can classify Indian temple
       images, but there remains scope for a feature-aware system that explicitly detects architectural components,
       evaluates whether explanations correspond to those components, and connects visual predictions to structured
       IKS architectural knowledge.


    6. Research Problem
    The project addresses four related problems:

•     Black-box classification: A model may produce a correct label without showing which architectural evidence it
      used.

•     Whole-image bias: Background, vegetation, sky, crowds or image-source characteristics can become unintended
      predictive cues.

•     Architectural complexity: Indian temple traditions contain regional and historical variations, so rigid categories can
      oversimplify the domain.

•     Lack of architectural explanation: A useful heritage system should answer why a classification was made in terms
      meaningful to an architectural researcher.



    7. Central Research Question
    Can a feature-aware and explainable deep-learning system identify Indian temple architectural traditions using
    architecturally meaningful visual characteristics rather than relying primarily on image-level or contextual
    correlations?


    8. Secondary Research Questions
•     Can deep-learning models distinguish selected architectural traditions from previously unseen temples?

•     Which architectural components contribute most strongly to classification?

•     Does explicit architectural component detection improve classification compared with a conventional whole-image
      CNN?

•     Can Explainable AI outputs be aligned with documented architectural features?

•     Does combining visual evidence with structured architectural knowledge improve interpretability?



IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                         Page 3
    9. Research Hypotheses
•     A feature-aware model incorporating architectural component information will provide more interpretable predictions
      than a conventional image-level classifier.

•     Explicit architectural feature detection will reduce dependence on irrelevant contextual regions.

•     Combining visual classification with structured architectural knowledge will improve explanation quality.



    10. IKS and Architectural Knowledge Foundation
    The IKS layer should be a real part of the research methodology rather than a historical introduction added only to
    satisfy the syllabus. The system should build a structured taxonomy of architectural components and link each
    component to reliable sources, definitions, visual characteristics and example monuments.

      Feature                       Role in the project                          Possible visual characteristics

      ■ikhara                       Northern-style superstructure analysis       Curvilinear/vertical tower forms in relevant traditions

      Vim■na                        Southern temple superstructure analysis      Stepped/pyramidal composition in relevant examples

      Gopura/Gopuram                Gateway detection                            Monumental gateway structures

      Ma■■apa                       Pillared-hall detection                      Pillared halls and roof structures

      Jagati / base                 Raised-platform analysis                     Elevated plinth/platform

      Amalaka                       Crowning-element analysis                    Ribbed circular crowning form in relevant examples

      Pillars                       Structural/decorative feature                Repeated vertical structural elements




    11. Scholarly Classification Caveat
    The paper should not claim that every Indian temple belongs cleanly to exactly one of N■gara, Dr■vi■a or Vesara. The
    terminology and boundaries of these categories, particularly Vesara, are debated in architectural scholarship. The
    project should therefore use carefully defined labels and explicitly document the classification scheme adopted for the
    dataset.

       A suitable dataset label can be “Deccan / traditionally classified Vesara” when the underlying sources support that
       terminology. The model should report a classification under the project's operational taxonomy rather than
       presenting it as an absolute historical truth.


    12. Proposed Solution
     Temple Image
     ■
     ■■■■■■■■■■■■■■■■■■■■■■■■■■■
     ↓ ↓
     Whole-image Encoder Component Detector
     ■ ■
     ↓ ↓
     Visual Features Architectural Features
     ■ ■
     ■■■■■■■■■■■■■■■■■■■■■■■■■■■
     ↓
     Feature Fusion
     ↓
     Style Classifier
     ↓
     Explainability Layer
     ↓
     IKS Knowledge Layer



IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                                         Page 4
     ↓
     Architectural Explanation



    13. Dataset Strategy
    The dataset should combine publicly available, appropriately licensed temple images with structured metadata.
    Potential sources include Wikimedia Commons, institutional digital heritage archives and public-domain or openly
    licensed collections. Every image should retain provenance and license information.

    Recommended metadata fields:

•     image_id

•     temple_id

•     temple_name

•     location

•     state/region

•     architectural_tradition

•     historical_period

•     dynasty where supported by the source

•     architectural_feature_labels

•     source

•     license

•     image_type



    14. Temple-Disjoint Evaluation
    This is one of the most important methodological requirements. Do not randomly split photographs if multiple
    photographs of the same temple exist. Otherwise, the model may memorize monument-specific visual characteristics.

     Training:
     Temple A, Temple B, Temple C, Temple D

     Testing:
     Temple E, Temple F, Temple G

     Rule:
     No temple appears in both training and test sets.

    A temple-disjoint protocol is also consistent with recent research methodology in Indian temple classification and
    provides a more realistic test of generalization.


    15. Architectural Component Dataset
    The second layer of the dataset should annotate architectural components. Depending on resources, annotations can
    be bounding boxes, segmentation masks or structured image-level labels.

•     ■ikhara

•     Vim■na

•     Gopuram




IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                           Page 5
•     Ma■■apa

•     Pillars

•     Jagati/base

•     Amalaka

•     Other project-specific components supported by the literature



    16. Baseline Models
    16.1 Conventional CNN Baseline
     Image
     ↓
     EfficientNet-B0
     ↓
     Global Average Pooling
     ↓
     Fully Connected Layer
     ↓
     Architectural Class

    EfficientNet-B0 is a useful baseline because recent Indian temple classification research has evaluated it directly.

    16.2 Lightweight Baseline
     Image
     ↓
     MobileNetV3
     ↓
     Classifier
     ↓
     Architectural Class

    MobileNetV3 can provide an accuracy-efficiency baseline and a pathway toward future mobile or edge deployment.


    17. Proposed Feature-Aware Model
     Temple Image
     ■
     ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
     ↓ ↓
     Whole-image CNN Component Detector
     ■ ■
     ↓ ↓
     Global Visual Features Architectural Features
     ■ ■
     ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
     ↓
     Feature Fusion
     ↓
     Classification Head
     ↓
     Architectural Style
     ↓
     Explainability + IKS

    The proposed model combines global visual representations with explicit architectural-component information. The
    exact fusion mechanism can be concatenation followed by fully connected layers, attention-based fusion, or another
    experimentally justified approach.


    18. Explainable AI

IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                             Page 6
 18.1 Grad-CAM
 Grad-CAM can visualize the image regions contributing to a CNN prediction. In this project, it should be evaluated
 against annotated architectural regions rather than presented only as a qualitative heatmap.

 18.2 Feature Attribution
 SHAP or another suitable attribution method can be used for interpretable feature contribution analysis, particularly for
 structured architectural features.

 18.3 Component-Level Explanation
 The target explanation is not merely “upper-left region was important.” Instead, the system should connect visual
 evidence to a detected architectural component such as a shikhara, vimana, gopuram or mandapa.


 19. Knowledge-Grounded Explanation
    Visual Evidence
    +
    Detected Architectural Components
    +
    Structured IKS Knowledge
    ↓
    Evidence-grounded explanation
    ↓
    Human-readable architectural report

 An optional LLM can convert structured evidence into natural language, but it should not independently decide the
 architectural class. The classification should come from the validated computer-vision pipeline and knowledge layer.


 20. Full System Architecture
    USER
    ■
    ↓
    Upload Temple Image
    ■
    ↓
    Image Preprocessing
    ■
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■
    ↓ ↓
    Whole-image CNN Architectural Detector
    ■ ■
    ↓ ↓
    Visual Features Component Features
    ■ ■
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■
    ↓
    Feature Fusion
    ↓
    Architectural Classifier
    ■
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■
    ↓ ↓
    Style Prediction Component Detection
    ■ ■
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■
    ↓
    Explainable AI
    ↓
    IKS Knowledge Base
    ↓
    Architectural Report




IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                        Page 7
    21. User Interface
    A practical prototype can be implemented as a React/TypeScript frontend with a FastAPI or Flask backend.

•     Upload temple image

•     Display predicted architectural tradition and confidence

•     Display detected architectural components

•     Show bounding boxes/segmentation masks

•     Show Grad-CAM or other explanation visualization

•     Display evidence-grounded architectural explanation

•     Display relevant source/provenance information



    22. Evaluation Metrics
      Category                           Metrics

      Classification                     Accuracy, Precision, Recall, Macro F1, Weighted F1

      Per-class performance              Confusion matrix, per-class Precision/Recall/F1

      Explainability                     Pointing Game, explanation-region IoU, occlusion/faithfulness tests

      Component detection                mAP, precision, recall, IoU; segmentation IoU/Dice if masks are used

      Robustness                         Performance under crops, lighting changes, occlusion and background perturbations

      Efficiency                         Inference time, parameter count and model size where relevant




    23. Explanation Evaluation
    A strong paper should evaluate whether the explanation corresponds to actual architectural evidence. One practical
    method is to compare explanation regions against human-annotated architectural components using Intersection over
    Union (IoU). Another is to mask the model's most important region and measure whether prediction confidence
    changes substantially.

    IoU: IoU = Area(A ∩ B) / Area(A ∪ B), where A is the explanation region and B is the annotated architectural
    component.


    24. Ablation Study
    Run at least four experiments:

      Experiment               Input/architecture

      A                        CNN baseline only

      B                        CNN + architectural components

      C                        CNN + components + structured knowledge

      D                        Full proposed model + explainability/evidence alignment



    Compare classification performance, generalization and explanation quality. Do not invent numerical results before
    experiments are conducted.




IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                               Page 8
    25. Robustness Experiments
•     Different lighting conditions

•     Different camera viewpoints

•     Partial occlusion

•     Tourists/people in foreground

•     Trees and sky in background

•     Low-resolution images

•     Cropped architectural regions

•     Background removal or masking



    26. Research Novelty
•     Feature-aware classification: combines whole-image features with explicit architectural components.

•     Architecture-specific XAI: explanations are tied to meaningful architectural structures.

•     Knowledge-grounded interpretation: visual evidence is mapped to a structured IKS architectural knowledge layer.

•     Temple-disjoint evaluation: tests generalization to unseen monuments.

•     Potential multimodal extension: combines visual information with textual architectural descriptions.



    27. Optional Multimodal Extension
     Image ■■→ Vision Encoder ■■■
     ■■■→ Multimodal Fusion → Classification
     Text ■■■→ Text Encoder ■■■■■■■

    The text input may contain a sourced architectural description. A multimodal model can test whether visual and textual
    evidence together improve recognition or explanation.


    28. Optional Knowledge Graph
     Temple
     ■
     ■■■ Location
     ■■■ Period
     ■■■ Dynasty
     ■■■ Architectural Tradition
     ■■■ ■ikhara Type
     ■■■ Vim■na
     ■■■ Ma■■apa
     ■■■ Gopuram
     ■■■ Source

    A knowledge graph can provide provenance-aware relationships among temples, architectural components, regions,
    periods and sources.


    29. Optional 3D Extension
    If the group has access to photogrammetry, LiDAR, drone imagery or smartphone depth scanning, the project can be
    extended toward 3D architectural-component segmentation and digital reconstruction. This should remain an extension
    rather than a requirement for the initial project because it substantially increases data and implementation complexity.



IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                         Page 9
    30. Technology Stack
     Layer                                   Recommended technologies

     Programming                             Python, TypeScript

     Deep learning                           PyTorch, torchvision

     Computer vision                         OpenCV, YOLO, SAM/SAM2 where justified

     Explainability                          Grad-CAM, SHAP, Integrated Gradients / occlusion analysis

     Data                                    Pandas, NumPy, scikit-learn

     Backend                                 FastAPI or Flask

     Frontend                                React, TypeScript, Tailwind CSS

     Database                                PostgreSQL/Supabase; Neo4j for knowledge graph if required

     Deployment                              Docker; Cloud Run/AWS/Azure as appropriate




    31. Development Timeline
     Weeks               Work

     1                   IKS and temple-architecture literature study; define taxonomy

     2                   Literature survey and research-gap analysis

     3–4                 Dataset collection, cleaning and metadata creation

     5                   EfficientNet/MobileNet baseline

     6–7                 Architectural component detection/segmentation

     8                   Feature-fusion model

     9                   Grad-CAM and explanation evaluation

     10                  Knowledge layer and provenance

     11                  Ablation, robustness and generalization experiments

     12                  React + FastAPI prototype

     13–14               Research paper, documentation and presentation




    32. Team Division
     Member                     Responsibilities

     Member 1                   IKS literature, architectural taxonomy, source verification, metadata

     Member 2                   Dataset construction, computer vision, component detection

     Member 3                   Feature fusion, XAI, experiments, ablation and evaluation

     Member 4                   Backend, frontend, database, visualization and deployment




    33. Expected Results
•    A model capable of classifying selected temple architectural traditions under a clearly defined taxonomy.




IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                   Page 10
•     Explicit identification of architectural components.

•     Visual and feature-level explanations for predictions.

•     Quantitative comparison with conventional image-level baselines.

•     Temple-disjoint evaluation demonstrating generalization to unseen monuments.

•     A searchable/provenance-aware architectural knowledge layer.

•     A working web prototype for demonstration and educational use.



    34. Expected Deliverables
•     Annotated Indian temple image dataset with provenance and licensing information.

•     Feature-aware temple architecture classifier.

•     Architectural component detector/segmenter.

•     Explainable AI module.

•     IKS architectural knowledge base or knowledge graph.

•     Web-based demonstration platform.

•     Research paper manuscript.

•     Potential software copyright or other IP assessment if the final implementation contains sufficiently original work.



    35. Publication Strategy
    The project should be positioned as a research contribution rather than as a generic application. Suitable venues
    depend on the final methodology and results and should be verified at submission time. Relevant areas include
    computer vision, digital cultural heritage, heritage informatics, AI and architectural computing.

    The goal should be a legitimate peer-reviewed venue whose scope matches the actual contribution. “Scopus” is an
    indexing database and should not be treated as a quality guarantee by itself.


    36. Suggested Five-Paper Literature Starting Set
      Research area                                           Why it matters to this project

      Indian temple architectural style recognition and interpretation
                                                                 Direct(2026)
                                                                        baseline: image classification, lightweight CNNs and Grad-CAM.

      3D semantic segmentation / reconstruction of Indian temple
                                                             Shows
                                                                 architecture
                                                                    AI-based(2026)
                                                                              component-level analysis and digital reconstruction.

      3DITA — 3D benchmark for Nagara-style Indian temple Provides
                                                          architecture
                                                                    a heritage point-cloud dataset direction.

      X-NeSyL / MonuMAI cultural-heritage work                Highly relevant to combining deep learning with expert knowledge graphs and explainability.

      Digital heritage documentation using imaging/scanning Provides
                                                           technologies
                                                                     context for computational preservation and documentation.




    37. Phase-1 250-Word Abstract
    Indian temple architecture represents a significant component of the Indian Knowledge System, incorporating
    structured principles of spatial organization, geometry, construction, ornamentation and architectural design. Traditional
    architectural scholarship describes diverse temple traditions including N■gara, Dr■vi■a and Deccan architectural
    forms, along with numerous regional and historical variations. However, identification and documentation of
    architectural styles from visual data still relies substantially on human expertise. This project proposes an Explainable


IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                                                 Page 11
 Multimodal Artificial Intelligence framework for automated recognition and interpretation of Indian temple architectural
 heritage. The proposed system will use computer vision and deep learning to classify temple architectural traditions
 while explicitly identifying meaningful architectural components such as shikhara, vimana, gopuram, mandapa, pillars
 and structural bases. Unlike conventional image-level classification systems, the proposed approach will combine
 whole-image visual representations with component-level architectural features and structured Indian architectural
 knowledge. Explainable AI techniques such as Grad-CAM and feature attribution will be employed to identify the visual
 regions and architectural components influencing model predictions. A temple-disjoint evaluation strategy will be
 adopted to measure generalization to previously unseen temples and reduce the possibility of monument-specific
 memorization. The project will further investigate whether incorporating architectural features improves classification
 performance and explanation quality compared with conventional deep-learning baselines. The expected outcome is a
 prototype platform capable of analyzing temple images, identifying architectural characteristics, providing interpretable
 predictions and presenting relevant knowledge about the detected architectural tradition. The project aims to contribute
 toward AI-assisted digital preservation, architectural education, heritage documentation and computational research in
 Indian Knowledge Systems.


 38. Recommended Final Research Framework
    INDIAN KNOWLEDGE SYSTEM
    ■
    ↓
    Temple Architecture
    ■
    ↓
    Architectural Knowledge Base
    ■
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    ↓ ↓
    Visual Dataset Textual Knowledge
    ■ ■
    ↓ ↓
    CNN / Vision Encoder Knowledge Encoder
    ■ ■
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    ↓
    Feature Fusion
    ↓
    Architectural Classifier
    ↓
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    ↓ ↓
    Style Prediction Component Detection
    ■ ■
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    ↓
    Explainable AI
    ↓
    Evidence Alignment
    ↓
    IKS Knowledge
    ↓
    Architectural Explanation
    ↓
    Web Application



 39. Conclusion
 The proposed project provides a strong intersection between Indian Knowledge Systems and modern computer
 science. Its strongest research direction is not simply automated temple-style classification, because that problem has
 already received recent research attention. The project should instead investigate whether explicit architectural
 components, explainability and structured IKS knowledge can make AI-based heritage analysis more reliable,
 interpretable and useful for previously unseen temples.




IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                      Page 12
    The recommended contribution is therefore: classification + architectural component detection + explanation
    alignment + IKS knowledge grounding + temple-disjoint evaluation. This combination provides a clear
    experimental methodology, measurable research questions, a working software prototype and a plausible path toward
    a peer-reviewed research paper.


    40. Selected References / Starting Sources
•     Indian temple architectural style recognition and interpretation using deep learning. Digital Applications in
      Archaeology and Cultural Heritage, 2026. Directly relevant baseline on temple image classification and Grad-CAM.

•     Deep learning-based semantic segmentation / 3D reconstruction research for Indian temple heritage. Measurement,
      2026. Relevant to 3D component-level heritage analysis.

•     3DITA — A 3D benchmark dataset for Nagara-style Indian temple architecture. ISPRS Archives, 2025.

•     X-NeSyL / MonuMAI cultural-heritage research on combining deep learning representations with expert knowledge
      graphs for explainability.

•     Digital heritage documentation research using scanning, photogrammetry and imaging technologies for Indian
      monuments.



    41. Important Research-Design Note
    Do not report accuracy, F1, IoU or other numerical results until the experiments have actually been performed. Do not
    claim that a model “understands” Indian architecture merely because it produces a plausible label. The paper should
    distinguish between model predictions, source-backed architectural knowledge and human interpretation. This
    separation will substantially improve the academic credibility of the project.




IKS Research Project — Explainable AI for Indian Temple Architectural Heritage                                       Page 13
