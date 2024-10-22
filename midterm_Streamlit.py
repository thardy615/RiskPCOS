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
    # Calculate correlations
    corr_matrix = subset.corr()

    # Display variables correlated with PCOS
    st.subheader(f"{title} Correlation with PCOS (Y/N)") # title
    pc_correlations = corr_matrix['PCOS (Y/N)'] # pcos class correlations
    for variable, value in pc_correlations.items(): # for each variable and it's respective correlation value with PCOS
        if variable != 'PCOS (Y/N)' and (value > 0.2 or value < -0.1): # If the correlation is < -0.1 but > 0.2,
            st.write(f"{variable}: {value:.2f}") # Print the variable and its correlation with PCOS

    # Display correlations between other variables (excluding PCOS correlations)
    st.subheader(f"Other {title} Variable Correlations")
    correlation_results = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] > 0.2 or corr_matrix.iloc[i, j] < -0.1) and \
                    (corr_matrix.columns[i] != 'PCOS (Y/N)' and corr_matrix.columns[j] != 'PCOS (Y/N)'): # If the correlation is < -0.1 but > 0.2 and if the variable of said correlation is not PCOS
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
    for col in numeric_columns: # for each numeric column
        if col in subset.columns and not subset[subset['PCOS (Y/N)'] == 0][col].empty: # For columns in subset, excluding target
            # Create box plot for the current numeric column
            fig = px.box(subset, x='PCOS (Y/N)', y=col, color='PCOS (Y/N)', title=f'{title} - {col} Boxplot', points='all')  # Show all points on the plot for better visualization
            # However, given outliers, I am considering removing. Consult with Murillo/Max later

            # Update layout
            fig.update_layout(xaxis_title='PCOS (Y/N)', yaxis_title=col, margin=dict(l=40, r=40, t=40, b=40)) # Includes (equal) margin space around plot area
            st.plotly_chart(fig)  # Plot

######################################################

resampled_data = prepare_resampled_data() # Use function above to get SMOTE dataframe

# Create subsets for visualizations for each page
hormone = resampled_data[['Age (yrs)', 'PCOS (Y/N)', 'FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 
 'PRG(ng/mL)', 'Pregnant(Y/N)']]
qualityOfLife = resampled_data[['Age (yrs)','PCOS (Y/N)',
 'Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
                   'Pimples(Y/N)', 'Reg.Exercise(Y/N)']]
metabolic = resampled_data[['Age (yrs)','PCOS (Y/N)', 'BMI', 'Waist:Hip Ratio', 'RBS(mg/dl)',
'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Pregnant(Y/N)', 'Reg.Exercise(Y/N)', 'Weight gain(Y/N)', 'Skin darkening (Y/N)']]
fertility = resampled_data[['Age (yrs)', 'PCOS (Y/N)', 'Cycle length(days)', 
'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 
                    'Avg. F size (R) (mm)', 'Endometrium (mm)', 'Pregnant(Y/N)', ]]


# Call the prepare_resampled_data function to prepare data and store in session state
# Just in case, since I kept running into trouble
if 'resampled_data' not in st.session_state:
    st.session_state.resampled_data = prepare_resampled_data()
    

# Sidebar navigation
st.sidebar.image(r"PCOS (1).png", use_column_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data", 'IDA/EDA: Hormone', 'IDA/EDA: Quality of Life', 'IDA/EDA: Metabolic', 'IDA/EDA: Fertility',"Principal Component Analysis", "Cluster Analysis", "Normal Lab Work Results", "Nomogram Risk Assessment"], index=0)


# Home Page (default)
if page == "Home":
    st.markdown("""<h1 style='color: pink;'><strong>RiskPCOS: A Polycystic Ovarian Syndrome (PCOS) Risk Assessment</h1>""", unsafe_allow_html=True)
    # Background Info
    st.markdown("""<p style="font-size:18px;">Polycystic Ovarian Syndrome, also known as PCOS, is a metabolic syndrome and hormonal condition that impacts the female reproductive system in women of reproductive age. Although every woman's experience differs, symptoms include irregular periods, hirsutism (excessive hair growth), insulin resistance, weight gain, male-patterned balding, acne, ovarian/follicular cysts, and infertility. PCOS directly impacts fertility by interfering with the growth and release of eggs from the ovaries. For diagnosis, patients stereotypically require at least 2 of the following criteria: irregular periods, high androgen levels, and ovarian cysts. According to the WHO, it is estimated that this condition affects 8-13% of women among reproductive age; however, 70% of cases go undiagnosed. Given the (lack of) care for women's reproductive health, it is very common for it to take years to diagnose women who do have it. 
    This app aims to predict PCOS diagnosis among fertile women and compare fertility measures (AMH) among infertile and fertile women with/without PCOS.</p>""", unsafe_allow_html=True)
    # Source Information
    st.write("Source: [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome)")
    st.markdown(""" <div style="color: black;"> Please venture through side bar options to learn more about the data used to assess PCOS risk, Initial Data Analysis, Exploratory Data Analysis, and the interactive Nomogram </div>""", unsafe_allow_html=True)
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
- BMI
- **Blood Group**
- Pulse rate (bpm)
- RR (breaths/min)
- Hb (g/dl)
- **Cycle (R/I)**
- Cycle length (days)
- **Marriage Status (Yrs)**
- **Pregnant (Y/N)**
- No. of abortions
- I beta-HCG (mIU/mL)
- II beta-HCG (mIU/mL)
- FSH (mIU/mL)
- LH (mIU/mL)
- FSH/LH
- Hip (inch)
- Waist (inch)
- Waist:Hip Ratio
- TSH (mIU/L)
- AMH (ng/mL)
- PRL (ng/mL)
- Vit D3 (ng/mL)
- PRG (ng/mL)
- RBS (mg/dl)
- **Weight gain (Y/N)**
- **Hair growth (Y/N)**
- **Skin darkening (Y/N)**
- **Hair loss (Y/N)**
- **Pimples (Y/N)**
- **Fast food (Y/N)**
- **Reg. Exercise (Y/N)**
- BP Systolic (mmHg)
- BP Diastolic (mmHg)
- Follicle No. (L)
- Follicle No. (R)
- Avg. F size (L) (mm)
- Avg. F size (R) (mm)
- Endometrium (mm)

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

    # Scale the numeric columns using z-score
    df_scaled = merged_df[true_numeric_cols].apply(zscore)

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

if page == 'IDA/EDA: Hormone':
    st.title("Hormone Analysis")
    st.subheader("Hormone Data")
    st.write(hormone)  # Display hormone data
    numeric_columns = ['Age (yrs)', 'FSH/LH', 'TSH (mIU/L)', 'AMH(ng/mL)', 
                           'PRL(ng/mL)', 'PRG(ng/mL)']
    categorical_columns = ['Pregnant(Y/N)']
    if st.button('Show Distributions'):
        plot_distributions(hormone, "Hormone", numeric_columns, categorical_columns)
    if st.button('Show Correlations'):
        plot_correlations(hormone, "Hormone")
    if st.button('Show Boxplots'):
        plot_boxplots(hormone, "Hormone", numeric_columns)

if page == 'IDA/EDA: Quality of Life':
    st.title("Quality of Life Analysis")
    st.subheader("Quality of Life Data")
    st.write(qualityOfLife)  # Display quality of life data
    numeric_columns = []
    categorical_columns = ['Pregnant(Y/N)', 'Weight gain(Y/N)', 
                               'hair growth(Y/N)', 'Skin darkening (Y/N)', 
                               'Hair loss(Y/N)', 'Pimples(Y/N)', 'Reg.Exercise(Y/N)']
    if st.button('Show Distributions'):
        plot_distributions(qualityOfLife, "Quality of Life", numeric_columns, categorical_columns)
    if st.button('Show Correlations'):
        plot_correlations(qualityOfLife, "Quality of Life")
    st.markdown(""" <br><br><div style="color: red;"> No numeric columns, so no boxplots </div>
""", unsafe_allow_html=True)

if page == 'IDA/EDA: Metabolic':
    st.title("Metabolic Analysis")
    st.subheader("Metabolic Data")
    st.write(metabolic)  # Display metabolic data
    numeric_columns = ['BMI', 'Waist:Hip Ratio', 'RBS(mg/dl)', 
                           'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)']
    categorical_columns = ['Pregnant(Y/N)', 'Reg.Exercise(Y/N)', 
                               'Weight gain(Y/N)', 'Skin darkening (Y/N)']
    if st.button('Show Distributions'):
        plot_distributions(metabolic, "Metabolic", numeric_columns, categorical_columns)
    if st.button('Show Correlations'):
        plot_correlations(metabolic, "Metabolic")
    if st.button('Show Boxplots'):
        plot_boxplots(metabolic, "Metabolic", numeric_columns)


if page == 'Fertility':
    st.title("Fertility Analysis")
    st.subheader("Fertility Data")
    st.write(fertility)  # Display fertility data
    numeric_columns = ['Cycle length(days)', 'Follicle No. (L)', 
                           'Follicle No. (R)', 'Avg. F size (L) (mm)', 
                           'Avg. F size (R) (mm)', 'Endometrium (mm)']
    categorical_columns = ['Pregnant(Y/N)']
    if st.button('Show Distributions'):
        plot_distributions(fertility, "Fertility", numeric_columns, categorical_columns)
    if st.button('Show Correlations'):
        plot_correlations(fertility, "Fertility")
    if st.button('Show Boxplots'):
        plot_boxplots(fertility, "Fertility", numeric_columns)
    
if page == 'Principal Component Analysis':
    st.title("Principal Component Analysis (PCA)")
    st.subheader("Coming Soon!")

if page == 'Cluster Analysis':
    st.title("Cluster Analysis")
    st.subheader("Coming Soon!")

if page == 'Nomogram Risk Assessment':
    st.title("Nomogram Risk Assessment")
    st.subheader("Coming Soon!")
    
if page == "Normal Lab Work Results":
    st.title("Normal Lab Work Results")
    st.subheader("Coming Soon!")
