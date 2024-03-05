import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.title('Mobile Price Range Prediction')                                                                   #Introduction
    image=Image.open(('phone_price_pred.jpg'))
    st.image(image,width=600)

    st.write('In this rapidly evolving technological landscape,we can see a lot of electronic devices,each of them offering a unique set of features and values. '
             'How can we accurately predict the price range of these gadgets. In the case of Mobile Phones, '
             'Price range prediction is essential to facilitate decision-making, '
             'empower consumers, and enable businesses to navigate the complexities of a rapidly evolving and diverse mobile market. '
             'Accurate price predictions assist consumers in planning their budgets effectively, aligning their preferences with '
             'available options and preventing unexpected financial burdens.')
    st.subheader('Objectives:')
    st.write('1. Diverse Mobile Market:\n The current generation witnesses a vast and diverse mobile market with a wide range of devices catering to '
             'different user needs and preferences.\n '
             '2. Evolving Technology:\n Rapid advancements in mobile technology lead to frequent releases of new models with varying features and '
             'capabilities.\n '
             '3. Affordability Concerns:\n Affordability is a significant concern for consumers, making accurate price predictions crucial for '
             'individuals seeking budget-friendly options.\n '
             '4. Consumer Empowerment:\n Mobile price prediction empowers consumers by providing them with information to make well-informed decisions, '
             'aligning with the tech-savvy nature of the current generation.\n '
             '5. Economic Considerations:\n Given economic fluctuations, consumers are more price-conscious, making accurate predictions valuable for '
             'budget planning.')

    st.subheader('Dataset')                                                                           #Dataframe
    st.write('Tap the button given below to see the dataset')
    data = st.button('Mobile price range prediction dataset')
    if data:
        st.write('This dataset contain collection of features characterizing mobile phones, including battery power, camera specifications, network support, '
                 'memory, screen dimensions, and other attributes. The price_range column categorizes phones into price ranges, '
                 'making this dataset suitable for mobile phone classification and price prediction tasks')
        df=pd.read_csv('mobile_price_prediction.csv')
        st.dataframe(df)

    st.subheader('Visualisation of Distribution of Classes')                                                  #Distribution of Classes
    def display_countplot():
        # Load data
        data = pd.read_csv('mobile_price_prediction.csv')
        # Create the countplot
        fig, ax = plt.subplots()
        sns.countplot(x='price_range', data=data, ax=ax)
        plt.xlabel('Price Range')
        plt.ylabel('Count')
        plt.title('Distribution of Mobile Phone Price Ranges')
        plt.figure(figsize=(1, 1))  # Adjust width and height as desired
        # Display the plot in Streamlit
        st.pyplot(fig)

    st.write('Tap the button given below to see the distribution of classes')
    dis = st.button('Plot')
    if dis:
        display_countplot()
        st.write('Conclusion: This is a multiclass classification dataset and all classes are equally distributed')


    st.subheader('Visualisation of Correlation matrix and Heatmap')
    st.write('Since the dataset contain 21 columns, executing feature selection for more accuracy')
    st.write('Tap the button to see the correlation matrix and heatmap')
    cor = st.button('Correlation Matrix')
    if cor:
        df=pd.read_csv('mobile_price_prediction.csv')                                                            #Correlation Matrix
        # Calculate correlation matrix
        corr_matrix = df.corr()
        # Display correlation matrix as a table (optional)
        st.subheader('Correlation Matrix')
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))  # Optional formatting

        st.subheader('Heatmap')                                                                                  #Heatmap
        # Create and display heatmap using st.pyplot
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm')  # Additional options for heatmap
        st.pyplot(fig)

        st.write('No conclusive insights can be drawn from the above heatmap. '
                 'The barplot below represents the values from the last column of the correlation matrix.')

                                                                                                            # Barplot of subset

        data = pd.read_csv('mobile_price_prediction.csv')
        y_column = 'price_range'  # Specify the column for plotting the correlation heatmap
        subset = corr_matrix[y_column]    # Extract the correlation values for the price_range column
        fig1=plt.figure(figsize=(10, 6))    # Plot the correlation values as a bar chart
        sns.barplot(x=subset.index, y=subset.values)
        plt.xlabel('Features of the Dataframe')
        plt.ylabel('Correlation')
        plt.title(f'Correlation with price_range')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig1)
        st.write('From the barplot, Dropped the column were correlation is nearly zero and then built the model')

    st.subheader('Accuracy Scores of models')                                                              #Accuracy Score of Models
    st.write('Model is built using several Classification Algorithms includes K Nearest '
             'Neighbors,Support Vector Classifier,Gaussian Naive Bayes, Decision Tree '
             'Classifier and Bagging Algorithms. If you want to see the accuracy '
             'scores of each algorithms, Tap the button given below ')
    img=st.button('Report')
    if img:
        image = Image.open(('Accuracy_Score.png'))                                                        #Classification Report of best model (SVC)
        st.image(image, width=600)
        st.write('Since support vector classifier(SVC()) gives more accuracy than rest,'
                 '85.3 is not a good accuracy , so after some tuning this is the report of SVC')
        image1=Image.open('SVC_report.png')
        st.image(image1,width=600)
        st.write('95.3 is a good accuracy score . so prediction is done by this model')





                                                   #PREDICTION




    st.subheader('Prediction')
    st.write('Correct inputs can lead to accurate results, so make sure '
             'the entered data is within the specified range. By agreeing '
             'to the terms and conditions, you may be directed to a prediction')
    agree = st.checkbox('I agree')
    if agree:
        te=st.slider('Energy Requirement of Battery',min_value=500,max_value=2000,step=1)
        bluet = st.radio('Bluetooth', ['Yes', 'No'])
        if bluet == 'Yes':
            bt = 1
        else:
            bt = 0

        duals = st.selectbox('Dual SIM Support',options=['Yes', 'No'],index=0)
        if duals == 'Yes':
            ds = 1
        else:
            ds = 0

        fc = st.slider('Front Camera Megapixels',min_value=0,max_value=20,step=1)

        fourg = st.selectbox('4G',options=['Yes', 'No'],index=0)
        if fourg=='Yes':
            frg=1
        else:
            frg=0

        im = st.slider('Internal Memory in GigaBytes',min_value=0,max_value=128,step=1)

        wt = st.slider('Weight of Mobile Phone (g)',min_value=80,max_value=200,step=1)

        pc = st.slider('Primary Camera mega pixels',min_value=2,max_value=25,step=1)

        ph = st.slider('Pixel Height', min_value=100,max_value=4000,step=10)

        pw = st.slider('Pixel Width ',min_value=100,max_value=4000,step=10)

        ram = st.slider('Random Access Memory in Mega Bytes',min_value=200,max_value=4000)

        tt = st.slider('Longest time that a single battery charge will last',min_value=5,max_value=25,step=1)

        threeg = st.selectbox('3G',options=['Yes', 'No'],index=0)
        if threeg == 'Yes':
            treg = 1
        else:
            treg = 0

        touch = st.radio('Touch Screen', ['Yes', 'No'])
        if touch == 'Yes':
            tc = 1
        else:
            tc = 0

        wifi = st.selectbox('WIFI',options=['Yes', 'No'],index=0)
        if wifi == 'Yes':
            wf = 1
        else:
            wf = 0



                                           #Model & Prediction


        features = [te, bt, ds, fc, frg, im, wt, pc, ph, pw, ram, tt, treg, tc, wf]


        model = pickle.load(open('mobile_price_model.sav', 'rb'))
        scaler = pickle.load(open('mobile_price_scaler.sav', 'rb'))

        pred = st.button('Prediction')
        if pred:
            prediction = model.predict(scaler.transform([features]))
            if prediction==0:
                st.write('Price Range: LOW')
            elif prediction==1:
                st.write('Price Range: MEDIUM')
            elif prediction==2:
                st.write('Price Range: HIGH')
            else:
                st.write('Price Range: VERY HIGH')


main()
