import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import r2_score
import joblib

### Creating functions ###

# Load and merge data
def load_data():
    fertile_df = pd.read_csv("pcos_data_fertile.csv")
    infertile_df = pd.read_csv("PCOS_infertility.csv")
    
    # Clean up column names for consistency
    fertile_df.columns = fertile_df.columns.str.strip()
    infertile_df.columns = infertile_df.columns.str.strip()

    # Convert columns to numeric where needed
    numeric_columns = ['II    beta-HCG(mIU/mL)', 'AMH(ng/mL)', 'Marraige Status (Yrs)', 'Fast food (Y/N)']
    for col in numeric_columns:
        fertile_df[col] = pd.to_numeric(fertile_df[col].astype(str).str.strip(), errors='coerce')
        infertile_df['AMH(ng/mL)'] = pd.to_numeric(infertile_df['AMH(ng/mL)'].astype(str).str.strip(), errors='coerce')

    # Merge dataframes
    merge_columns = ['Sl. No', 'Patient File No.', 'PCOS (Y/N)', 'I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'AMH(ng/mL)']
    merged_df = pd.merge(fertile_df, infertile_df, on=merge_columns, how='left')
    
    return merged_df

# Use SMOTE to get equally distributed (50/50 non-PCOS and PCOS data) resampled data
def prepare_resampled_data():
    # Load data
    merged_df = load_data()
    merged_df = merged_df.apply(pd.to_numeric, errors='coerce') # Coerce to numeric columns
    merged_df = merged_df.dropna() # Drop NAs
    # Decided to do this after assessing missingness^
    
    true_numeric_cols = ['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Pulse rate(bpm)', 'RR (breaths/min)', 'Hb(g/dl)', 'Cycle length(days)', 
                     'Marraige Status (Yrs)', 'No. of aborptions','I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 
                     'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio', 
                     'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)',
                     'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 
                    'Avg. F size (R) (mm)', 'Endometrium (mm)']
    df_scaled = merged_df[true_numeric_cols] # this is not scaled for right now, but will be when regression model is produced
    non_scaled_cols = ['Sl. No', 'Patient File No.', 'PCOS (Y/N)', 'Blood Group', 'Cycle(R/I)', 'Pregnant(Y/N)', 
                   'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
                   'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)']
    df_final = pd.concat([df_scaled, merged_df[non_scaled_cols]], axis=1)
    
    # separate features (X) and target (y)
    X = df_final.drop('PCOS (Y/N)', axis=1) # data without 'PCOS (Y/N)' column
    y = df_final['PCOS (Y/N)']
    y = y.loc[X.index]  # makes sure to only keep rows in y that match X
    
    # execute SMOTE
    smote = SMOTE(random_state=42) # sets seed
    X_resampled, y_resampled = smote.fit_resample(X, y) # resamples the data and returns the X array containing resampled data and their corresponding labels

    resampled_data = pd.DataFrame(X_resampled, columns=X.columns) # recreate data frame
    resampled_data['PCOS (Y/N)'] = y_resampled # Get target column in new data frame
    return resampled_data #return data frame after SMOTE, which should now have even data for PCOS and non-PCOS
    # Code above was sourced from Murillo's code in homework #5

# Visualize missing values in heatmap
def visualize_missing_values(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns # Get numeric columns
    data_subset = data[numeric_cols] # subset data with only numeric columns

    nan_mask = data_subset.isna().astype(int).to_numpy() # create a boolean mask: True for NaN, False for finite values; convert boolean mask to integer (False becomes 0, True becomes 1)

    plt.figure(figsize=(12, 6)) # size the plot
    plt.imshow(nan_mask.T, interpolation='nearest', aspect='auto', cmap='viridis') # imshow with interpolation set to 'nearest' and aspect to 'auto'
    plt.xlabel('Patient Index')
    plt.ylabel('Features')
    plt.title('Visualizing Missing Values in Data')
    plt.yticks(range(len(data_subset.columns)), data_subset.columns)
    plt.xticks(np.linspace(0, nan_mask.shape[0]-1, min(10, nan_mask.shape[0])).astype(int))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    st.pyplot(plt) # Plot
    # This code was sourced from Murillo's code in Homework #4

# Correlation plots   
def plot_correlations(subset, title):
    st.write("Note: Utilizes scaled data")
    # Calculate correlations
    corr_matrix = subset.corr()

    # Display variables correlated with PCOS
    st.subheader(f"{title} Correlation with PCOS (Y/N)") # title
    pc_correlations = corr_matrix['PCOS (Y/N)'] # pcos class correlations
    for variable, value in pc_correlations.items(): # for each variable and it's respective correlation value with PCOS
        if variable != 'PCOS (Y/N)' and (value > 0.2 or value < -0.2): # If the correlation is < -0.2 but > 0.2,
            st.write(f"{variable}: {value:.2f}") # Print the variable and its correlation with PCOS

    # Display correlations between other variables (excluding PCOS correlations)
    st.subheader(f"Other {title} Variable Correlations")
    correlation_results = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] > 0.2 or corr_matrix.iloc[i, j] < -0.2) and \
                    (corr_matrix.columns[i] != 'PCOS (Y/N)' and corr_matrix.columns[j] != 'PCOS (Y/N)'): # If the correlation is < -0.2 but > 0.2 and if the variable of said correlation is not PCOS
                correlation_results.append(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}") # Print the variable and its correlation with another
    # Display correlation results in app
    for result in correlation_results:
        st.write(result)

    # interactive heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmin=-1, zmax=1, hoverongaps=False))

    # Update layout
    fig.update_layout(title=f'{title} Correlation Matrix', xaxis_nticks=36, margin=dict(l=40, r=40, t=40, b=40), height=600, width=800)
    st.plotly_chart(fig) # Plot
# ChatGPT 4o was utilized (for corrections on code in midterm.ipynb since streamlit was not utilized in that file) for the above code on 10/19/24

# Histogram Plots    
def plot_distributions(subset, title, numeric_columns, categorical_columns):
    # Plot numeric columns as interactive histograms with rug plots
    st.write("Note: Utilizes unscaled data")
    for col in numeric_columns: # for each numeric column
        if col in subset.columns and not subset[subset['PCOS (Y/N)'] == 0][col].empty: # For columns in subset, excluding target
            # Create histograms for Non-PCOS and PCOS
            fig = ff.create_distplot(
                [subset[subset['PCOS (Y/N)'] == 0][col], subset[subset['PCOS (Y/N)'] == 1][col]], # Separate data for Non-PCOS and PCOS groups 
                group_labels=['Non-PCOS', 'PCOS'], show_hist=True, show_rug=True, bin_size='auto') # Include labels (targets), create both a rug plot and histogram, and automatically determine bin size
            # Add title and labels
            fig.update_layout(title=f'{title} - {col} Distribution', xaxis_title=col, yaxis_title='Density', legend_title='PCOS (Y/N)',margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig)  # plot

    # Plot categorical columns as interactive bar plots
    for col in categorical_columns: # for each numeric column
        if col in subset.columns and not subset[col].empty: # For columns in subset, excluding target
            # Create histograms for Non-PCOS and PCOS
            fig = px.histogram( subset, x=col, color='PCOS (Y/N)', barmode='group', title=f'{title} - {col} Distribution') # Title is based on column/variable name
            # Update layout
            fig.update_layout(xaxis_title=col, yaxis_title='Count', legend_title='PCOS (Y/N)', margin=dict(l=40, r=40, t=40, b=40)) # Includes (equal) margin space around plot area
            st.plotly_chart(fig)  # Plot

# Box Plots
def plot_boxplots(subset, title, numeric_columns):
    # Plot numeric columns as interactive box plots
    st.write("Note: Utilizes scaled data")
    for col in numeric_columns: # for each numeric column
        if col in subset.columns and not subset[subset['PCOS (Y/N)'] == 0][col].empty: # For columns in subset, excluding target
            # Create box plot for the current numeric column
            fig = px.box(subset, x='PCOS (Y/N)', y=col, color='PCOS (Y/N)', title=f'{title} - {col} Boxplot', points='all')  # Show all points on the plot for better visualization
            # However, given outliers, I am considering removing. Consult with Murillo/Max later

            # Update layout
            fig.update_layout(xaxis_title='PCOS (Y/N)', yaxis_title=col, margin=dict(l=40, r=40, t=40, b=40)) # Includes (equal) margin space around plot area
            st.plotly_chart(fig)  # Plot

# Function to plot confusion matrix
def plot_confusion_matrix(model_name, y_true, y_pred):
    st.write("Note: Utilizes scaled data")
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, cmap='Blues', colorbar=False
    )
    ax.set_title(f"Confusion Matrix: {model_name}")
    st.pyplot(fig)
    
# Define a function to calculate the "log-odds" based on the model's decision function
def calculate_risk(features_unscaled, model, scaler, numeric_features):
    # Separate numeric and binary features
    numeric_inputs_unscaled = [features_unscaled[feature] for feature in numeric_features]
    binary_inputs = [features_unscaled[feature] for feature in features_unscaled if feature not in numeric_features]
    
    # Scale numeric inputs
    numeric_inputs_scaled = scaler.transform([numeric_inputs_unscaled])[0]
    
    # Combine scaled numeric inputs with binary inputs
    model_inputs = []
    for feature in features_unscaled:
        if feature in numeric_features:
            model_inputs.append(numeric_inputs_scaled[numeric_features.index(feature)])
        else:
            model_inputs.append(features_unscaled[feature])
    
    # Calculate log-odds using the model's decision function
    decision_value = model.decision_function([model_inputs])  # Log-odds
    risk = 1 / (1 + np.exp(-decision_value))  # Convert log-odds to probability
    return risk[0]

######################################################

resampled_data = prepare_resampled_data() # Use function above to get SMOTE dataframe

# Call the prepare_resampled_data function to prepare data and store in session state
# Just in case, since I kept running into trouble
if 'resampled_data' not in st.session_state:
    st.session_state.resampled_data = prepare_resampled_data()
    
# Create subsets for visualizations for each page
hormone_unscaled = resampled_data[['Age (yrs)', 'PCOS (Y/N)', 'FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 
 'PRG(ng/mL)', 'Pregnant(Y/N)']]
qualityOfLife_unscaled = resampled_data[['Age (yrs)','PCOS (Y/N)',
 'Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
                   'Pimples(Y/N)', 'Reg.Exercise(Y/N)']]
metabolic_unscaled = resampled_data[['Age (yrs)','PCOS (Y/N)', 'BMI', 'Waist:Hip Ratio', 'RBS(mg/dl)',
'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Pregnant(Y/N)', 'Reg.Exercise(Y/N)', 'Weight gain(Y/N)', 'Skin darkening (Y/N)']]
fertility_unscaled = resampled_data[['Age (yrs)', 'PCOS (Y/N)', 'Cycle length(days)', 
'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 
                    'Avg. F size (R) (mm)', 'Endometrium (mm)', 'Pregnant(Y/N)', ]]


### Variables that are correlated with PCOS
true_numeric_cols = ['BMI','Follicle No. (L)', 'Follicle No. (R)']
# Define columns to log scale
log_scale_cols = ['AMH(ng/mL)']

# Scale the remaining numeric columns using z-score
remaining_cols = [col for col in true_numeric_cols if col not in log_scale_cols]
data_scaled = resampled_data[remaining_cols].apply(zscore)

# Combine the log-scaled columns and z-score scaled columns
# Apply log scaling to the specified columns
data_scaled[log_scale_cols] = resampled_data[log_scale_cols].apply(lambda x: np.log1p(x))
# data_scaled[log_scale_cols] = resampled_data[log_scale_cols]

# Non-scaled columns
non_scaled_cols = [
    'Sl. No', 'Patient File No.', 'PCOS (Y/N)', 
    'Blood Group', 'Cycle(R/I)', 'Pregnant(Y/N)', 
    'Weight gain(Y/N)', 'hair growth(Y/N)', 
    'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
    'Pimples(Y/N)', 'Fast food (Y/N)', 
    'Reg.Exercise(Y/N)'
]

# Combine scaled and non-scaled columns to create final df
# final_model_data = pd.concat([df_scaled, resampled_data[non_scaled_cols]], axis=1)

# Scaling
# data_scaled = resampled_data[true_numeric_cols].apply(zscore)
non_scaled_cols = ['hair growth(Y/N)', 'Skin darkening (Y/N)', 'Pimples(Y/N)','Weight gain(Y/N)', 'PCOS (Y/N)']
final_model_data = pd.concat([data_scaled, resampled_data[non_scaled_cols]], axis=1)
features = ['BMI', 'AMH(ng/mL)', 'Follicle No. (L)', 'Follicle No. (R)', 
                   'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Pimples(Y/N)','Weight gain(Y/N)', 'PCOS (Y/N)']
    

# Sidebar navigation
st.sidebar.image(r"PCOS (1).png", use_column_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data", 'IDA/EDA: Hormone', 'IDA/EDA: Quality of Life', 'IDA/EDA: Metabolic', 'IDA/EDA: Fertility',"Principal Component Analysis", "Models", "Nomogram Risk Assessment"], index=0)


# Home Page (default)
if page == "Home":
    st.markdown("""<h1 style='color: pink;'><strong>RiskPCOS: A Polycystic Ovarian Syndrome (PCOS) Risk Assessment</h1>""", unsafe_allow_html=True)
    # Background Info
    st.markdown("""<p style="font-size:18px;">Polycystic Ovarian Syndrome, also known as PCOS, is a metabolic syndrome and hormonal condition that impacts the female reproductive system in women of reproductive age. PCOS directly impacts fertility by interfering with the growth and release of eggs from the ovaries. </p>""", unsafe_allow_html=True)
    st.write(""" #### Although every woman's experience differs, symptoms include:
    
    - **irregular periods**
    - **hirsutism (excessive hair growth)**
    - **insulin resistance**
    - **weight gain**
    - **brain fog**
    - **anxiety and/or depression**
    - **male-patterned balding**
    - **(cystic) acne**
    - **ovarian/follicular cysts**
    - **infertility**
    """)
    
    st.write(""" #### For diagnosis, patients stereotypically require at least 2 of the following criteria:
    
    - **irregular periods**
    - **high androgen levels**
    - **ovarian cysts**

    """)
    st.markdown("""<p style="font-size:18px;"> According to the World Health Organization(WHO), it is estimated that this condition affects 8-13% of women among reproductive age; however, 70% of cases go undiagnosed. Given the (lack of) care for women's reproductive health, it is very common for it to take years to diagnose women who do have it.</p>""", unsafe_allow_html=True)
    # Display an image using a URL
    image_url = "https://www.nishantivfcare.com/wp-content/uploads/2023/12/Nishant-Blog-Banner-9-min.jpg"
    st.image(image_url, use_column_width=True)
    # Source Information
    st.write("Source: [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome)")
    st.write(""" #### The RiskPCOS app aims to predict PCOS diagnosis among fertile and infertile women. Here's how:
    
    - Clean publically available data found suitable for generating predictions (page `Data`).
    - Explore the data within each variable and how they correlate to PCOS (all `IDA/EDA` pages).
        - Due to the high number of variables being evaluated, I found it best to separate them into categories so that correlations would be easy to calculate and easy for users to view. Additionally, If condensed onto 1 page, the page would be extremely long and thus, possibly overwhelming for viewers.
    - Execute Principal Component Analysis (PCA) for reducing dimensionality of data prior to making predictions (page `Principal Component Analysis`).
    - Generate and evaluate multiple models/algorithms to determine which would be best for predicting PCOS (page `Models`).
    - Utilize the most accurate model in a nomogram, allowing viewers to vary variables to access PCOS risk (page `Nomogram`).""")
    
    st.markdown(""" <div style="color: black;"> Please venture through side bar options (in order or appearance) to learn more about the data used to assess PCOS risk. </div>""", unsafe_allow_html=True)
# Disclaimer
    st.markdown(""" <br><br><div style="color: red;"> **Disclaimer:** I am not a medical practitioner, so neither I nor this app can officially diagnose anyone with Polycystic Ovarian Syndrome. This app is useful for risk assessment only. Viewers who may consider themselves at risk can gather their findings and take them to their primary care physician, OB/GYN, or endocrinologist to retrieve an actual diagnosis and thus (hopefully) receive treatment. For more information about PCOS, please visit the WHO link above or investigate other reliable online sources, but it is recommended to speak to a medical provider. </div>
""", unsafe_allow_html=True)

# Data Page:
elif page == "Data":
    st.markdown("""<h1 style='color: pink;'><strong>Data Source and Data Manipulation </h1>""", unsafe_allow_html=True)
    # Footer about data
    st.markdown("""
<p style="font-size:18px;">For my app creation, I am using PCOS data consisting of fertile and infertile patients from Kaggle: https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos/data . 
</p>""", unsafe_allow_html=True)

    st.markdown("""
The clinical data for both the infertile and fertile datasets were collected across 10 hospitals in India and includes the following variables (variables in **bold** have label encoding):

- **PCOS (Y/N) (TARGET)**
- Age (yrs)
- Weight (Kg)
- Height (Cm)
- BMI: Body Mass Index (weight (kg) / (height (m))^2)
- **Blood Group**
- Pulse rate (bpm): Heart rate
- RR (breaths/min): Respiratory Rate
- Hb (g/dl): Hemoglobin
- **Cycle (R/I)**: Regular/Irregular Cycle
- Menstrual Cycle length (days)
- **Marriage Status (Yrs)**
- **Pregnant (Y/N)**
- No. of abortions
- I beta-HCG (mIU/mL): 1st variation of beta-Human Chorionic Gonadotropin (hCG) hormone to detect pregnancy, and can also be used as a tumor marker
- II beta-HCG (mIU/mL): 2nd variation of beta-Human Chorionic Gonadotropin (hCG) hormone to detect pregnancy, and can also be used as a tumor marker
- FSH (mIU/mL): Follicle-stimulating hormone is a hormone that stimulates egg growth in women during their menstrual cycle
- LH (mIU/mL): Luteinizing hormone is a hormone produced in the pituitary gland that stimulates the ovaries to release eggs and produce other hormones
- FSH/LH: Ratio between LH and FSH usually lies between 1-2. For women with PCOS, this ratio becomes reversed, and it might reach as high as 2 or 3, resulting in ovulation not occurring.
- Hip (inch)
- Waist (inch)
- Waist:Hip Ratio: Ratio to assess the distribution of fat on your body. Higher ratios can mean patients carry more fat around their waist
- TSH (mIU/L): Thyroid-stimulating hormone is produced by the pituitary gland and regulates the production of hormones by the thyroid gland
- AMH (ng/mL): Anti-mullerian hormone assesses a woman's ovarian reserve or egg count. This hormone is produced by the small follicles in a woman's ovaries.
- PRL (ng/mL): Prolactin (also known as lactotropin) is a hormone that's responsible for lactation. This hormone is expected to be high in pregnant/lactating women. 
- Vit D3 (ng/mL): Vitamin D
- PRG (ng/mL): Progesterone is a hormone that supports menstruation and pregnancy. Normally, it rises during pregnancy and while using some birth control medications.
- RBS (mg/dl): Fasting Blood Glucose Test
- **Weight gain (Y/N)**
- **Excessive Hair growth (Y/N)**: Hirsutism
- **Skin darkening (Y/N)**
- **Hair loss (Y/N)**: Male-patterned baldness
- **Pimples (Y/N)**
- **Fast food (Y/N)**
- **Reg. Exercise (Y/N)**
- BP Systolic (mmHg): Systolic Blood Pressure
- BP Diastolic (mmHg): Diastolic Blood Pressure
- Follicle No. (L): Number of follicular cysts on left ovary (caused by anovulation)
- Follicle No. (R): Number of follicular cysts on right ovary (caused by anovulation)
- Avg. F size (L) (mm): Average size of follicular cysts on left ovary
- Avg. F size (R) (mm): follicular cysts on right ovary
- Endometrium (mm): The tissue that lines the uterus. This organ thickens in preparation for pregnancy, and if a fertilized egg implants, the lining remains in place for the fetus. If pregnancy doesn't occur, the endometrium sheds during the menstrual cycle.

**One limitation this data contains is a lack of information on androgen/male hormones, which includes DHEA, DHEAS, and Free/Bioavailable/Total Testosterone. I believe this could be because of the variability of these hormones throughout the (on average, 28 day) cycle, so the hospitals I have obtained the data from may not have been able to include this data due to inconsistent values and possibly, unreliability.**

**Androgens are a diagnostic criteria for PCOS, so I believe obtaining this information in the future would greatly improve my prediction model.**

Before any data manipulation, missingingness and class/sub-class sizes need to be accessed. Overall, the data only contained a few missing values (Missing Completely at Random aka MCAR), so simply removing them does not interfere with how the data is observed. However, about 1/3 of patients are classified to have PCOS while 2/3 of patients are not, so SMOTE is used to ensure equal class distribution between non-PCOS and PCOS patients. How I processed my data is broken down below.
""", unsafe_allow_html=True)

    ### Note: Had to redo prepare_resampled_data() and visualize_missing_values() here to break it down on the data page
    ### I kept getting errors intially doing it only here since this code block below and above is page specific & not on the entire app
    # Load data
    merged_df = load_data()
    st.write("Initial (Merged) DataFrame Shape:", merged_df.shape)
    st.write(merged_df.head()) # Display first few rows

    # Capture the output of df.info() in a string buffer
    buffer = io.StringIO()
    merged_df.info(buf=buffer)
    info_str = buffer.getvalue()  # Get the string content of the buffer
    # Display df.info() using Streamlit
    st.write("Variable information:")
    st.text(info_str)  # Use st.text() for plain text display
    # Display df.describe() directly using Streamlit
    st.write(""" #### Summary Statistics for all variables:""")
    st.write(merged_df.describe())
    # The chunk of code above was sourced from ChatGPT 4o on 10/20/2024

    # Visualize missing values
    st.subheader("Visualization/Heatmap of Missing Values in Dataframe to Access Missingness")
    visualize_missing_values(merged_df)
    plt.clf()
    merged_df = merged_df.dropna()  # Drop any rows with NA values

    # Define numeric columns to scale
    true_numeric_cols = [
    'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 
    'Pulse rate(bpm)', 'RR (breaths/min)', 'Hb(g/dl)', 
    'Cycle length(days)', 'Marraige Status (Yrs)', 
    'No. of aborptions', 'I   beta-HCG(mIU/mL)', 
    'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 
    'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio', 
    'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 
    'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)',
    'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 
    'Follicle No. (L)', 'Follicle No. (R)', 
    'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 
    'Endometrium (mm)'
]

# Define columns to log scale
    log_scale_cols = ['FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRG(ng/mL)']

# Apply log scaling to the specified columns
    merged_df[log_scale_cols] = merged_df[log_scale_cols].apply(lambda x: np.log1p(x))

# Scale the remaining numeric columns using z-score
    remaining_cols = [col for col in true_numeric_cols if col not in log_scale_cols]
    df_scaled = merged_df[remaining_cols].apply(zscore)

# Combine the log-scaled columns and z-score scaled columns
    df_scaled[log_scale_cols] = merged_df[log_scale_cols]

# Non-scaled columns
    non_scaled_cols = [
    'Sl. No', 'Patient File No.', 'PCOS (Y/N)', 
    'Blood Group', 'Cycle(R/I)', 'Pregnant(Y/N)', 
    'Weight gain(Y/N)', 'hair growth(Y/N)', 
    'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
    'Pimples(Y/N)', 'Fast food (Y/N)', 
    'Reg.Exercise(Y/N)'
]

# Combine scaled and non-scaled columns to create final df
    df_final = pd.concat([df_scaled, merged_df[non_scaled_cols]], axis=1)

    # Visualize missing values after removal
    st.subheader("Missing Values Visualization After Missingness Removal")
    visualize_missing_values(df_final)
    # Clear the current figure to avoid overlap
    plt.clf()

    # Display class counts
    class_counts = df_final['PCOS (Y/N)'].value_counts()
    st.write("Class Distribution:")
    st.write(class_counts)

    # Percentage of PCOS cases
    percentage = (class_counts[1] / class_counts.sum()) * 100
    st.write(f"Percentage of PCOS cases: {percentage:.2f}%")
    

    st.markdown("<br>", unsafe_allow_html=True)  # Add another break for spacing

    # SMOTE
    X = df_final.drop('PCOS (Y/N)', axis=1)
    y = df_final['PCOS (Y/N)']
    # Bar chart for class distribution before SMOTE
    st.subheader('Class Distribution Before SMOTE')
    sns.countplot(x=y)
    plt.title('Class Distribution Before SMOTE')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    st.pyplot(plt)
    # Clear the current figure to avoid overlap
    plt.clf()
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

    smote = SMOTE(random_state=42) # sets seed
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_data['PCOS (Y/N)'] = y_resampled
    all_variables_scaled = resampled_data
    st.session_state.all_variables_scaled = all_variables_scaled
    

    # Display class distribution after SMOTE
    st.write("Class distribution after SMOTE:")
    st.write(pd.Series(y_resampled).value_counts())

    # Show final DataFrame shape after resampling
    st.write("Final DataFrame Shape after SMOTE (not including target column):", X_resampled.shape)

    # Separate the visualizations with some space
    st.markdown("<br>", unsafe_allow_html=True)  # Add a break for spacing

    # Bar chart for class distribution after SMOTE
    st.subheader('Class Distribution After SMOTE')
    sns.countplot(x=y_resampled)
    plt.title('Class Distribution After SMOTE')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.markdown("<br>", unsafe_allow_html=True)  # Add another break for spacing


if 'all_variables_scaled' in st.session_state:
    all_variables_scaled = st.session_state.all_variables_scaled
else:
    st.write("To retrieve scaled data, go to Data page before viewing IDA/EDA")
# Create subsets for visualizations for each page
hormone = all_variables_scaled[['Age (yrs)', 'PCOS (Y/N)', 'FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 
 'PRG(ng/mL)', 'Pregnant(Y/N)']]
qualityOfLife = all_variables_scaled[['Age (yrs)','PCOS (Y/N)',
 'Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
                   'Pimples(Y/N)', 'Reg.Exercise(Y/N)']]
metabolic = all_variables_scaled[['Age (yrs)','PCOS (Y/N)', 'BMI', 'Waist:Hip Ratio', 'RBS(mg/dl)',
'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Pregnant(Y/N)', 'Reg.Exercise(Y/N)', 'Weight gain(Y/N)', 'Skin darkening (Y/N)']]
fertility = all_variables_scaled[['Age (yrs)', 'PCOS (Y/N)', 'Cycle length(days)', 
'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 
                    'Avg. F size (R) (mm)', 'Endometrium (mm)', 'Pregnant(Y/N)', ]]
if page == 'IDA/EDA: Hormone':
    st.markdown("""<h1 style='color: pink;'><strong>Hormone Variables' IDA/EDA </h1>""", unsafe_allow_html=True)
    # Footer about data
    st.markdown("""
<p style="font-size:18px;">The table below includes my variables used for this IDA/EDA. I generated Distribution Plots, Correlations, and Box Plots for each variable in respect to patients have PCOS (0 = No, 1 = Yes). For variables correlated with PCOS, those variables were utilized in my PCA analysis and my models accessed for the nomogram.  
</p>""", unsafe_allow_html=True)
    st.write(hormone_unscaled)  # Display hormone data
    numeric_columns = ['Age (yrs)', 'FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 
                           'PRL(ng/mL)', 'PRG(ng/mL)']
    categorical_columns = ['Pregnant(Y/N)']
    if st.button('Show Distributions'):
        plot_distributions(hormone_unscaled, "Hormone", numeric_columns, categorical_columns)
    if st.button('Show Correlations'):
        plot_correlations(hormone, "Hormone")
        st.write("AMH determines egg reserve, and those with PCOS tend to have a higher egg reserve due to not ovulating/menstruating. This checks out!")
    if st.button('Show Boxplots'):
        plot_boxplots(hormone, "Hormone", numeric_columns)

if page == 'IDA/EDA: Quality of Life':
    st.markdown("""<h1 style='color: pink;'><strong>'Quality of Life' Variables' IDA/EDA </h1>""", unsafe_allow_html=True)
    # Footer about data
    st.markdown("""
<p style="font-size:18px;">The table below includes my variables used for this IDA/EDA. I generated Distribution Plots, Correlations, and Box Plots for each variable in respect to patients have PCOS (0 = No, 1 = Yes). For variables correlated with PCOS, those variables were utilized in my PCA analysis and my models accessed for the nomogram.   
</p>""", unsafe_allow_html=True)
    st.subheader("Quality of Life Data")
    st.write(qualityOfLife_unscaled)  # Display quality of life data
    numeric_columns = []
    categorical_columns = ['Pregnant(Y/N)', 'Weight gain(Y/N)', 
                               'hair growth(Y/N)', 'Skin darkening (Y/N)', 
                               'Hair loss(Y/N)', 'Pimples(Y/N)', 'Reg.Exercise(Y/N)']
    if st.button('Show Distributions'):
        plot_distributions(qualityOfLife_unscaled, "Quality of Life", numeric_columns, categorical_columns)
    if st.button('Show Correlations'):
        plot_correlations(qualityOfLife, "Quality of Life")
        st.write("Insulin resistance is a symptom (and possibly cause) of PCOS, so it is safe to assume that strong correlations were expected for weight gain and skin darkening.")
    st.markdown(""" <br><br><div style="color: red;"> No numeric columns, so no boxplots </div>
""", unsafe_allow_html=True)

if page == 'IDA/EDA: Metabolic':
    st.markdown("""<h1 style='color: pink;'><strong>Metabolic Variables' IDA/EDA </h1>""", unsafe_allow_html=True)
    # Footer about data
    st.markdown("""
<p style="font-size:18px;">The table below includes my variables used for this IDA/EDA. I generated Distribution Plots, Correlations, and Box Plots for each variable in respect to patients have PCOS (0 = No, 1 = Yes). For variables correlated with PCOS, those variables were utilized in my PCA analysis and my models accessed for the nomogram.   
</p>""", unsafe_allow_html=True)
    st.subheader("Metabolic Data")
    st.write(metabolic_unscaled)  # Display metabolic data
    numeric_columns = ['BMI', 'Waist:Hip Ratio', 'RBS(mg/dl)', 
                           'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)']
    categorical_columns = ['Pregnant(Y/N)', 'Reg.Exercise(Y/N)', 
                               'Weight gain(Y/N)', 'Skin darkening (Y/N)']
    if st.button('Show Distributions'):
        plot_distributions(metabolic_unscaled, "Metabolic", numeric_columns, categorical_columns)
    if st.button('Show Correlations'):
        plot_correlations(metabolic, "Metabolic")
        st.write("Insulin resistance is a symptom (and possibly cause) of PCOS, so it is safe to assume that strong correlations were expected for BMI.")

    if st.button('Show Boxplots'):
        plot_boxplots(metabolic, "Metabolic", numeric_columns)


if page == 'IDA/EDA: Fertility':
    st.markdown("""<h1 style='color: pink;'><strong>Fertility Variables' IDA/EDA </h1>""", unsafe_allow_html=True)
    # Footer about data
    st.markdown("""
<p style="font-size:18px;">The table below includes my variables used for this IDA/EDA. I generated Distribution Plots, Correlations, and Box Plots for each variable in respect to patients have PCOS (0 = No, 1 = Yes). For variables correlated with PCOS, those variables were utilized in my PCA analysis and my models accessed for the nomogram.   
</p>""", unsafe_allow_html=True)
    st.subheader("Fertility Data")
    st.write(fertility_unscaled)  # Display fertility data
    numeric_columns = ['Cycle length(days)', 'Follicle No. (L)', 
                           'Follicle No. (R)', 'Avg. F size (L) (mm)', 
                           'Avg. F size (R) (mm)', 'Endometrium (mm)']
    categorical_columns = ['Pregnant(Y/N)']
    if st.button('Show Distributions'):
        plot_distributions(fertility_unscaled, "Fertility", numeric_columns, categorical_columns)
    if st.button('Show Correlations'):
        plot_correlations(fertility, "Fertility")
        st.write("Follicular cysts are one of the diagnostic criteria for PCOS, so it is safe to assume that strong correlations were expected for the number of follicles in either ovary.")

    if st.button('Show Boxplots'):
        plot_boxplots(fertility, "Fertility", numeric_columns)

# Correlation Analysis revealed the following variables correlate with PCOS:
# Age, AMH, Pregnant (Y/N), Weight Gain(Y/N), Hair Growth (Y/N), Skin Darkening (Y/N), Pimples (Y/N), BMI, Cycle length(days), Follicle No. (L), Follicle No. (R)
# These variables will be used in further analysis.

if page == 'Principal Component Analysis':
    st.markdown("""<h1 style='color: pink;'><strong>Principal Component Analysis (PCA) </h1>""", unsafe_allow_html=True)
    # Display a brief introduction or description
    st.subheader("Exploring dimensionality reduction using PCA.")
    st.write("""
        Principal Component Analysis (PCA) helps in reducing the dimensionality of data 
        while retaining most of the variance. Below, you can interact with the PCA plot 
        and visualize the relationships between the variables in the transformed space Additionally, the sidebar allows users to include all 8 variables or exclude up to 6 of your choosing. Keep in mind, if only 2 PCs are chosen, the 3D plot with not execute.
    """)
    st.write(final_model_data)  # Display data being used in PCA (11 variables + target)

    # Sidebar for interactivity
    st.sidebar.header("PCA Configuration")
    selected_features = st.sidebar.multiselect(
        "Select features for PCA",
        features, 
        default=features[:-1]  # Removes the last feature ('PCOS (Y/N)') from the default selection
    )
    color_by = st.sidebar.selectbox("Color by:", ['PCOS (Y/N)'])

    # Ensure features are selected
    if len(selected_features) < 2:
        st.error("Please select at least two features for PCA.")
    else:
        scaled_data = final_model_data[selected_features].values
        pca = PCA()
        components = pca.fit_transform(scaled_data)

        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_ * 100
        labels = {str(i): f"PC {i+1}" for i in range(len(selected_features))}
        final_model_data[color_by] = final_model_data[color_by].astype(str)
        

        # Create scatter matrix plot
        fig = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(min(13, len(explained_variance))),  # Show up to 13 PCs
            color=final_model_data[color_by],
            color_discrete_map={'1': 'pink', '0': 'blue'} 
        )
        fig.update_traces(diagonal_visible=True)
        fig.update_layout(font=dict(size=8),xaxis=dict(tickangle=0), yaxis=dict(tickangle=0))

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        # Show explained variance 
        st.write("Explained (Default) Variance Percentages:")
        bullet_points = "\n".join([f"- **{selected_features[i]}/PC{i + 1}**: {var:.2f}%" for i, var in enumerate(pca.explained_variance_ratio_[:len(selected_features)])])
        st.markdown(bullet_points)
        
    # Add a 3D PCA visualization
    st.subheader("Interactive 3D PCA Plot")
    st.write("""I have also visualized the data in a 3D principal component space to explore clustering or patterns 
    associated with PCOS classification. Viewing the default 3D plot, it is clear that patients with PCOS are distinguishable from patients who do not have PCOS""")

    # Ensure at least 3 components are available for 3D plotting
    if components.shape[1] >= 3:
        # Create a DataFrame for plotting
        pca_df = pd.DataFrame(
            {
                'PC1': components[:, 0],
                'PC2': components[:, 1],
                'PC3': components[:, 2],
                'PCOS (Y/N)': final_model_data[color_by]
        }
    )
    
        # Create the 3D scatter plot
        fig_3d = px.scatter_3d(
            pca_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='PCOS (Y/N)',
            color_discrete_map={'1': 'pink', '0': 'blue'},
            title="3D PCA Plot",
            labels={
                'PC1': f"PC1 ({explained_variance[0]:.1f}%)",
                'PC2': f"PC2 ({explained_variance[1]:.1f}%)",
                'PC3': f"PC3 ({explained_variance[2]:.1f}%)"
            },
            template="plotly_white"
        )
    
        # Customize plot appearance
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title=f"PC1 ({explained_variance[0]:.1f}%)"),
                yaxis=dict(title=f"PC2 ({explained_variance[1]:.1f}%)"),
                zaxis=dict(title=f"PC3 ({explained_variance[2]:.1f}%)")
            ),
            width=1000,
            height=800
        )

        # Display the 3D PCA plot in Streamlit
        st.plotly_chart(fig_3d)
    else:
        st.warning("Not enough components available for a 3D PCA plot. Select more features.")

if page == 'Models':
    st.markdown("""<h1 style='color: pink;'><strong>Machine Learning Models That Can Predict PCOS Risk </h1>""", unsafe_allow_html=True)
    st.subheader("Exploring Different Models for Classification")
    st.write("""
### Model Selection Summary:
The table below includes the variables utilized in PCA, which showed strong correlations with PCOS. I have explored and trained the following models:

- **Linear Regression**: Finds the linear relationship between variables to make predictions.
- **Logistic Regression**: Models the log-odds of an event as a linear combination of independent variables to make predictions.
- **LASSO Regression**: aka "Least Absolute Shrinkage and Selection Operator," performs both variable selection and shrinkage by adding a penalty term to the standard regression model, which allows shrinking some coefficients to zero selecting the most relevant features.
- **Support Vector Machines (SVM)**: Algorithm that distinguishes features between classes in target by finding the best hyperplane that maximizes the margin between the closest data points of these classes.
  - Kernels: 'linear', 'rbf', 'poly', 'sigmoid'
- **Naive Bayes**: An algorithm that assumes that all features are completely independent of each other when predicting the target.

The model with the **best accuracy** is the **SVM model** using the **rbf (Radial Basis Function)** kernel.

To interpet the model performance, I have generated a confusion matrix (comparing its predicted labels to the actual labels) for each model. Additionally, I have included the R² value for each model. R² evaluates the goodness of fit by calculating the variance explained by the model divided by the total variance.
An R² of 0% means the model does not explain any of the variation in the response variables around their respective means, meanwhile 100% means that all variation is explained. Keep in mind, it is possible for a good model to have a low R², and it is possible for a biased/unfit model to have a high R².
""")
    # Display an image using a URL
    image_url2 = "https://cdn.prod.website-files.com/660ef16a9e0687d9cc27474a/662c42677529a0f4e97e4f96_644aea65cefe35380f198a5a_class_guide_cm08.png"
    st.image(image_url2, caption = "How to interpret a confusion matrix for a machine learning model", use_column_width=True)
    st.write("Displaying the data utilized in each model for reference:")
    st.write(final_model_data) # Display dataset for reference

    # Split the data into training and test data
    target = 'PCOS (Y/N)'
    X = final_model_data.drop(columns=[target])  # Features
    y = final_model_data[target].astype(int)     # Target (binary)
    split_pct = 0.70
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_pct, random_state=42)

    # Models
    results = {}

    ## Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    accuracy_lin = accuracy_score(y_test, y_pred_lin.round())  # Round predictions
    r2_lin = r2_score(y_test, y_pred_lin)
    results['Linear Regression'] = (accuracy_lin, r2_lin)

    ## Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    accuracy_log = accuracy_score(y_test, y_pred_log)
    r2_log = r2_score(y_test, y_pred_log)
    results['Logistic Regression'] = (accuracy_log, r2_log)

    ## LASSO Regression
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    accuracy_lasso = accuracy_score(y_test, y_pred_lasso.round())  # Round predictions
    r2_lasso = r2_score(y_test, y_pred_lasso)
    results['LASSO Regression'] = (accuracy_lasso, r2_lasso)

    ## Support Vector Machines (SVM)
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    svm_accuracies = {}
    svm_r2_scores = {}
    for kernel in kernels:
        svm_model = SVC(kernel=kernel, random_state=42)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        svm_accuracies[kernel] = accuracy_score(y_test, y_pred_svm)
        svm_r2_scores[kernel] = r2_score(y_test, y_pred_svm)

    best_svm_kernel = max(svm_accuracies, key=svm_accuracies.get)
    results['SVM (Best Kernel)'] = (svm_accuracies[best_svm_kernel], svm_r2_scores[best_svm_kernel])
    best_svm_model = SVC(kernel=best_svm_kernel, random_state=42)
    if 'best_svm_model' not in st.session_state:
        st.session_state.best_svm_model = best_svm_model
        best_svm_model.fit(X_train, y_train)
        st.write("Model has been trained and stored in session state.")
    else:
        print("Model already exists in session state.")
    best_svm_model.fit(X_train, y_train)

    ## Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    r2_nb = r2_score(y_test, y_pred_nb)
    results['Naive Bayes'] = (accuracy_nb, r2_nb)

    # Display results
    st.subheader("Model Comparisons:")
    for model, (acc, r2) in results.items():
        st.subheader(f"{model} Accuracy: {acc:.2f}, R²: {r2:.2f}")
    
        # Generate predictions for the corresponding model
        if model == "Linear Regression":
            y_pred = lin_reg.predict(X_test).round()
        elif model == "Logistic Regression":
            y_pred = log_reg.predict(X_test)
        elif model == "LASSO Regression":
            y_pred = lasso.predict(X_test).round()
        elif model.startswith("SVM"):
            y_pred = best_svm_model.predict(X_test)
            st.write(f"Best SVM Kernel: {best_svm_kernel}")
        elif model == "Naive Bayes":
            y_pred = nb_model.predict(X_test)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(1.75, 1.75))
        ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues", colorbar=False
    )
        ax.set_title(f"Confusion Matrix: {model}")
        st.pyplot(fig)
    
if page == 'Nomogram Risk Assessment':
    st.markdown("""<h1 style='color: pink;'><strong> Interactive Nomogram for PCOS Risk Prediction </h1>""", unsafe_allow_html=True)
    if 'best_svm_model' in st.session_state:
        best_svm_model = st.session_state.best_svm_model
    else:
        st.write("Model not found. Please load or train the model first.")
    # Description of the tool
    st.subheader("Adjust the following features to predict the risk of PCOS")
    st.write("""This nomogram allows you to adjust the values of different features, 
    and based on the selected `best_svm_model`, the risk of having PCOS will be calculated.""")

    target_variable = 'PCOS (Y/N)'  # Replace with the actual name of your target variable

    # Exclude the target variable from the features list
    numeric_features = [feature for feature in features if feature != target_variable and len(final_model_data[feature].unique()) > 2]
    binary_features = [feature for feature in features if feature != target_variable and feature not in numeric_features]

    scaler = StandardScaler()
    scaler.fit(resampled_data[numeric_features])

    feature_inputs_unscaled = {}

    # Sliders for numeric features (display unscaled values)
    for idx, feature in enumerate(numeric_features):
        min_val = resampled_data[feature].min()
        max_val = resampled_data[feature].max()
        mean_val = resampled_data[feature].mean()

        # Check that all values passed to the slider are float types (if they are not already)
        min_val = float(min_val)
        max_val = float(max_val)
        mean_val = float(mean_val)
        
        # Add the slider input for the user
        feature_inputs_unscaled[feature] = st.slider(
            f"Adjust {feature}", min_value=min_val, max_value=max_val, value=mean_val
        )

    # Dropdowns for binary features
    for feature in binary_features:
        feature_inputs_unscaled[feature] = st.selectbox(
            f"Select {feature}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Calculate the risk based on the model

    risk = calculate_risk(feature_inputs_unscaled, best_svm_model, scaler, numeric_features)
    
    # Display the calculated risk as a percentage
    st.subheader(f"Estimated Risk of PCOS: {risk * 100:.2f}%")

    st.write("In the future I hope to include androgen hormone measurements in my model. Additionally, I would like to access even more different types of models to see if I can improve my nomogram!")

    

