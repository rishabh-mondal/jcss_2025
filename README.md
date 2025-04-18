# **🌍 Space to Policy: Scalable Brick Kiln Detection and Automatic Compliance Monitoring with Geospatial Data**

📄 [**Space to Policy: Scalable Brick Kiln Detection and Automatic Compliance Monitoring with Geospatial Data**](https://arxiv.org/pdf/2412.04065) — *Patel et al.*

🌐 Explore the interactive project page: [Brick-Kilns](https://sustainability-lab.github.io/brick-kilns)


## 📋 Outline

- [Data Details](#-data-details)
- [Code Details](#-code-details)
- [Figures](#-figures)

---

## 📊 Data Details

1. **Satellite Imagery – Planet Labs**

   - 📥 [**Download Quads of Planet Imagery**](data_details/download_planet_quads.ipynb)  
     Notebook for downloading PlanetScope imagery tiles (quads) for selected regions.

   - ⚙️ [**Preprocessing & Label Generation**](data_details/data_and_label_preprocessing.ipynb)  
     Converts Planet imagery quads into usable inputs and generates YOLO-format labels for detection.


---

## 🧠 Code Details

1. **Training and Evaluation**


    - 🏋️ [**Training Models**](code_details/runner.sh)  
    Train different models on the initial dataset.

    - 📊 [**Table 3 — Model Performance**](code_details/map_numbers.ipynb)  
    Performance of various models on the initial dataset, including mAP (Mean Average Precision) calculations.

2. **Compliance Monitoring**

    - 🏛️ [**Table 5 — Compliance Monitoring**](code_details/compliance_monitoring.ipynb)  
    Automatic compliance detection of brick kilns across states based on state-wise and central environmental policies.

3. **Emission**     

    - 🌫️ [**Table 6 — Emission Rates by Kiln Type**](code_details/table-emission_rates.ipynb)  
    Emission rates (in g/kg of fired brick) for different kiln technologies, based on prior studies.

    - 🌍 [**Table 7 — State-wise Production & Emissions**](code_details/table-emission_rates.ipynb)  
    Daily brick production and estimated emissions (in tonnes) for each state.


---


## 📈 Figures

1. **Spatial Distribution**

    - 🗺️ [**Fig. 4 — Kiln Locations Across States**](figure_details/figure-brick-kiln-locations.ipynb)  
    Brick kiln locations across five states of the Indo-Gangetic Plain, India.

2. **Survey Comparison**

    - 🧮 [**Fig. 5 — District-wise Kiln Counts in Uttar Pradesh**](figure_details/figure-up_comparison_with_survey.ipynb)  
    Comparison of district-wise brick kiln counts in Uttar Pradesh:  
    (a) UPPCB 2023 survey vs. (b) our hand-validated data.

3. **Temporal Analysis**

    - 🔄 [**Fig. 6 — Kiln Technology Evolution (2010–2022)**](figure_details/brick_kilns_conversion.ipynb)  
    Brick kiln technology transition over 12 years for two NCAP non-attainment city airsheds:  
    (a) Delhi Airshed and (b) Lucknow Airshed.

