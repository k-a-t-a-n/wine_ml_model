import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier  
st.set_page_config(page_title = "wine analysis",page_icon ='grapes',layout ='wide',initial_sidebar_state='expanded' )

rouge = pd.read_csv('https://raw.githubusercontent.com/k-a-t-a-n/wine_ml_model/main/winequality-red.csv', sep = ';')
blanc = pd.read_csv('https://raw.githubusercontent.com/k-a-t-a-n/wine_ml_model/main/winequality-white.csv', sep = ';')
complet = pd.read_csv('https://raw.githubusercontent.com/k-a-t-a-n/wine_ml_model/main/wine_complete.csv', sep = ',')
couleurs = complet

couleurs['color_code'] = np.where(couleurs['color']=='white', '#ffffff', '#E3D081')


#   cd C:\Users\alecs\Desktop\CODE\vin 
#   streamlit run vin.py
st.title('Can a grade be given to wine with machine learning?')



st.write("Select a wine color on the left and take a look at the dataframe.")




color = complet['color'].unique()
color_select = st.sidebar.radio('Select a wine color:', color)
filtered_data = complet[complet['color'] == color_select]

st.write(filtered_data)

col1, col2= st.beta_columns([2, 2])

with col1:
    st.write(" ")
    st.write(" ")
    st.subheader("""Let's start with a heatmap and focus on "*__quality__*". """)
    st.write(" ")

with col2:
    fig, ax = plt.subplots()

    
    plt.style.use("dark_background")
    sns.heatmap(filtered_data.corr(),cmap="rocket_r", center= 0, ax=ax)
    sns.cubehelix_palette(as_cmap=True)
    st.write(fig)

with col1:
    
    st.write("It seems that none of the column will be of any help to find a correlation on its own.")
    st.write(" ")



st.write(" ")

with st.beta_container():
    st.subheader("Let's try with a *positive* and a *negative* correlation.")

#fig2
    fig, axes = plt.subplots(1,2, figsize=(15, 5), sharey=True)
    sns.regplot(ax=axes[0],x='alcohol', y='quality', data=filtered_data,scatter_kws={'color':'#8C2C3F'},line_kws={"color": "crimson"})
    axes[0].set_title('Positive correlation')
    sns.regplot(ax=axes[1],x='density', y='quality', data=filtered_data,scatter_kws={'color':'#8C2C3F'},line_kws={"color": "crimson"})
    axes[1].set_title("Negative correlation")

    st.write(fig)


    st.write("A denser wine will have a bad grade and one with more alcohol will taste better. But is it still enough? From here, we will rather try a machine learning model (knn) to figure out how exactly the composition of the wine is responsible for it's grade.")

#ML

X = filtered_data[['fixed acidity',
         'volatile acidity',
         'citric acid',
         'residual sugar',
         'chlorides',
         'free sulfur dioxide',
         'total sulfur dioxide',
         'density',
         'pH',
        'sulphates',
        'alcohol']]
y= filtered_data['quality']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=36, train_size = 0.75)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X, y)


st.write(" ")
st.write(f'the model we use gives us an accuracy score of : **{model.score(X_train, y_train)}** on the train sample and an accuracy score of : **{model.score(X_test, y_test)}** on the test sample. The score means we have a solid base but it still is not enough')
st.write(" ")


st.write("let's try one final test, try out your favorite wine, give its composition below and see the grade the model would give to it and compare it with your personal grade")



fixed_acidity = st.slider("what is the wine's fixed_acidity ",2.0,20.0, step = 0.1)
volatile_acidity = st.slider("what is the wine's volatile_acidity ", 0.0 ,2.0, step = 0.01)
citric_acid = st.slider("what is the wine's citric_acid ",0.0 ,2.0, step = 0.01)
residual_sugar = st.slider("what is the wine's residual_sugar ",0.0 ,70.0, step = 0.1)
chlorides = st.slider("what is the wine's chlorides ",0.0 ,0.7, step = 0.001)
free_sulphure_dioxide = st.slider("what is the wine's free_sulphure_dioxide ",1.0 ,300.0, step = 0.1)
total_sulphure_dioxide = st.slider("what is the wine's total_sulphure_dioxide ",5 ,450, step = 1)
density = st.slider("what is the wine's density ",0.90000 ,1.10000, step = 0.00001)
pH = st.slider("what is the wine's pH ",1.0 ,5.0, step = 0.1)
sulphates = st.slider("what is the wine's sulphates ",0.0 ,3.0, step = 0.01)
alcohol = st.slider("what is the wine's alcohol ",6.0 ,16.0, step = 0.01)
    
my_data = np.array([fixed_acidity ,
                    volatile_acidity, 
                    citric_acid,
                    residual_sugar,
                    chlorides,
                    free_sulphure_dioxide,
                    total_sulphure_dioxide,
                    density,
                    pH,
                   sulphates,
                   alcohol]).reshape(1,11)
note = model.predict(my_data)
    
st.subheader(f"This wine's grade is : {note}")     

st.write(" ")
st.write("You can change the number as much as you want, it definitly will not give you a perfect grade, there are two reasons for this : ")
st.write(" ")
st.write("- The first is that no wine had a grade of 10 in the base dataset, so it's the limit of this study.")
st.write("- The second reason is that the composition of the wine is not sufficient enough to gauge it's quality, many more factors need to be take in account.")


