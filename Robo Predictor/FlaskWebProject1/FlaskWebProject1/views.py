"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template,request,redirect
from FlaskWebProject1 import app
from wtforms import TextField
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
#import matplotlib.pyplot as plt
# Data cleanup
#x = ('C:/Repositories/Hospital General Information.csv');

df=pd.read_csv('C:/Repositories/Hospital General Information.csv', encoding = "ISO-8859-1")
df1=df.copy()
#print (df["Hospital overall rating"].value_counts())
#df["Hospital overall rating"] = df["Hospital overall rating"].fillna(df["Hospital overall rating"].mode())
df1["Hospital overall rating"][df1["Hospital overall rating"]=='Not Available']=df1["Hospital overall rating"].mode()[0]
trainingData=df[["Hospital overall rating","Patient experience national comparison","Hospital Type","Hospital Ownership","Emergency Services","Meets criteria for meaningful use of EHRs" ,"Mortality national comparison", "Safety of care national comparison"]]
#print (df1["Hospital overall rating"].value_counts())
#df1["Hospital overall rating"] = df1["Hospital overall rating"].fillna(df1["Hospital overall rating"].mode())
#print(df1["Hospital overall rating"].mode()[0])
#print (df1["Patient experience national comparison"].value_counts())
#Not Available                   
#Above the national average      
#Same as the national average    
#Below the national average      

df1["Patient experience national comparison"][df1["Patient experience national comparison"]=='Above the national average']=1
df1["Patient experience national comparison"][df1["Patient experience national comparison"]=='Same as the national average']=2
df1["Patient experience national comparison"][df1["Patient experience national comparison"]=='Below the national average']=3
df1["Patient experience national comparison"][df1["Patient experience national comparison"]=='Not Available']=4
print (df1["Patient experience national comparison"].value_counts())

#print (df1["Hospital Type"].value_counts())
#Acute Care Hospitals         3369
#Critical Access Hospitals    1344
#Childrens                      99
df1["Hospital Type"][df1["Hospital Type"]=='Acute Care Hospitals']=1
df1["Hospital Type"][df1["Hospital Type"]=='Critical Access Hospitals']=2
df1["Hospital Type"][df1["Hospital Type"]=='Childrens']=3
#print (df1["Hospital Type"].value_counts())

#print (df1["Hospital Ownership"].value_counts())
#Voluntary non-profit - Private                 2052
#Proprietary                                     800
#Government - Hospital District or Authority     561
#Voluntary non-profit - Other                    462
#Government - Local                              407
#Voluntary non-profit - Church                   343
#Physician                                        68
#Government - State                               65
#Government - Federal                             45
#Tribal                                            9
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Voluntary non-profit - Private']=1
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Proprietary']=2
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Government - Hospital District or Authority']=3
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Voluntary non-profit - Other']=4
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Government - Local']=5
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Voluntary non-profit - Church']=6
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Physician']=7
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Government - State']=8
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Government - Federal']=9
df1["Hospital Ownership"][df1["Hospital Ownership"]=='Tribal']=10
#print (df1["Hospital Ownership"].value_counts())
#print (df1["Emergency Services"].value_counts())
#Yes    4497
#No      315
df1["Emergency Services"][df1["Emergency Services"]=='Yes']=1
df1["Emergency Services"][df1["Emergency Services"]=='No']=2
#print (df1["Emergency Services"].value_counts())
#print (df1["Meets criteria for meaningful use of EHRs"].value_counts())
#print (df1["Mortality national comparison"].value_counts())
#Same as the national average    2719
#Not Available                   1352
#Above the national average       400
#Below the national average       341
df1["Mortality national comparison"][df1["Mortality national comparison"]=='Same as the national average']=1
df1["Mortality national comparison"][df1["Mortality national comparison"]=='Above the national average']=2
df1["Mortality national comparison"][df1["Mortality national comparison"]=='Below the national average']=3
df1["Mortality national comparison"][df1["Mortality national comparison"]=='Not Available']=4
#print (df1["Mortality national comparison"].value_counts())
#print (df1["Safety of care national comparison"].value_counts())
#Not Available                   2168
#Same as the national average    1194
#Above the national average       786
#Below the national average       664
df1["Safety of care national comparison"][df1["Safety of care national comparison"]=='Same as the national average']=1
df1["Safety of care national comparison"][df1["Safety of care national comparison"]=='Above the national average']=2
df1["Safety of care national comparison"][df1["Safety of care national comparison"]=='Below the national average']=3
df1["Safety of care national comparison"][df1["Safety of care national comparison"]=='Not Available']=4
#print (df1["Safety of care national comparison"].value_counts())
#print (df1["Timeliness of care national comparison"].value_counts())
df1["Timeliness of care national comparison"][df1["Timeliness of care national comparison"]=='Same as the national average']=1
df1["Timeliness of care national comparison"][df1["Timeliness of care national comparison"]=='Above the national average']=2
df1["Timeliness of care national comparison"][df1["Timeliness of care national comparison"]=='Below the national average']=3
df1["Timeliness of care national comparison"][df1["Timeliness of care national comparison"]=='Not Available']=4

#print (df1["Efficient use of medical imaging national comparison"].value_counts())
df1["Efficient use of medical imaging national comparison"][df1["Efficient use of medical imaging national comparison"]=='Same as the national average']=1
df1["Efficient use of medical imaging national comparison"][df1["Efficient use of medical imaging national comparison"]=='Above the national average']=2
df1["Efficient use of medical imaging national comparison"][df1["Efficient use of medical imaging national comparison"]=='Below the national average']=3
df1["Efficient use of medical imaging national comparison"][df1["Efficient use of medical imaging national comparison"]=='Not Available']=4

trainingData=df1[["Provider ID","Hospital Name","ZIP Code","Hospital overall rating","Patient experience national comparison","Hospital Type","Hospital Ownership","Emergency Services","Mortality national comparison", "Safety of care national comparison","Timeliness of care national comparison","Efficient use of medical imaging national comparison"]]
#print (trainingData.head())
#print ("length of data",len(trainingData))
from sklearn.model_selection import train_test_split
train,test = train_test_split(trainingData,test_size = 0.15)
#print ("length of training data",len(train))
#print ("length of testing data",len(test))
target=train["Hospital overall rating"].values
features=train[["Patient experience national comparison","Hospital Type","ZIP Code","Hospital Ownership","Mortality national comparison", "Safety of care national comparison","Timeliness of care national comparison","Efficient use of medical imaging national comparison"]]
my_tree = RandomForestClassifier(max_depth = 15, min_samples_split=2,n_estimators =100, random_state = 1)
my_tree = my_tree.fit(features,target)
gradient=GradientBoostingClassifier(random_state = 1)
gradient=gradient.fit(features,target)
#newX = selectKImportance(my_tree,features,5)
#print("feature importance",(my_tree.feature_importances_))
#print ("Training accuracy:",(my_tree.score(features,target)))
training_accuracy = my_tree.score(features,target)
test_features=test[["Patient experience national comparison","Hospital Type","ZIP Code","Hospital Ownership","Mortality national comparison", "Safety of care national comparison","Timeliness of care national comparison","Efficient use of medical imaging national comparison"]].values
my_prediction = my_tree.predict(test_features)
gradient_predict=gradient.predict(test_features)
from sklearn.metrics import accuracy_score
#print ("prediction accuracy:",accuracy_score(test["Hospital overall rating"],my_prediction))
testing_accuracy=accuracy_score(test["Hospital overall rating"],my_prediction)
gradient_accuracy=accuracy_score(test["Hospital overall rating"],gradient_predict)
print ("prediction accuracy:",gradient_accuracy)
#print (df1["Provider ID"].value_counts())
import bokeh.charts as bc
from bokeh.resources import CDN
from bokeh.embed import components
# Create the plot
#@app.route("/hospital_vis")
#def hospital_vis():
# # Build the dataframe
# datanew=df["Hospital overall rating"].value_counts()
# # Create the plot
# plot = bc.Line(title='Count of Hospital rating!',
# data=datanew, xlabel='Count', ylabel='Hospital Rating')

# # Generate the script and HTML for the plot
# script, div = components(plot)

# # Return the webpage
# return """
#<!doctype html>
#<head>
# <title>My wonderful trigonometric webpage</title>
# {bokeh_css}
#</head>
#<body>
# <h1>Everyone loves trig!
# {div}

# {bokeh_js}
# {script}
#</body>
# """.format(script=script, div=div, bokeh_css=CDN.render_css(),
# bokeh_js=CDN.render_js())


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )



@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )


@app.route('/hospital1', methods = ['GET','POST'])
def hospital1():
    data = {} 
    if request.form:
        # get the form data
        form_data = request.form
        data['form'] = form_data
        predict_patient_experience = form_data['predict_patient_experience']
        predict_zip_code = float(form_data['predict_zip_code'])
        predict_hospital_type = form_data['predict_hospital_type']
        predict_ownership = form_data['predict_ownership']
        predict_services = form_data['predict_services']
        predict_EHR = form_data['predict_EHR']
        predict_mortality = form_data['predict_mortality']
        predict_care = form_data['predict_care']
        predict_timeliness = form_data['predict_timeliness']
        predict_imaging = form_data['predict_imaging']
        #print(data)
        if predict_patient_experience == 'Above the national average':
            predict_patient_experience = 1
        elif predict_patient_experience == 'Same as the national average':
            predict_patient_experience = 2
        elif predict_patient_experience == 'Below the national average':
            predict_patient_experience = 3
        else:
            predict_patient_experience = 4

        if predict_hospital_type == 'Acute Care Hospitals':
            predict_hospital_type = 1
        elif predict_hospital_type == 'Critical Access Hospitals':
            predict_hospital_type = 2
        else:
            predict_hospital_type = 3

        if predict_ownership == 'Voluntary non-profit - Private':
            predict_ownership = 1
        elif predict_ownership == 'Proprietary':
            predict_ownership = 2
        elif predict_ownership == 'Government - Hospital District or Authority':
            predict_ownership = 3
        elif predict_ownership == 'Voluntary non-profit - Other':
            predict_ownership = 4
        elif predict_ownership == 'Government - Local':
            predict_ownership = 5
        elif predict_ownership == 'Voluntary non-profit - Church':
            predict_ownership = 6
        elif predict_ownership == 'Physician':
            predict_ownership = 7
        elif predict_ownership == 'Government - State':
            predict_ownership = 8
        elif predict_ownership == 'Government - Federal':
            predict_ownership = 9
        else:
            predict_ownership = 10

        if predict_services == 'Yes':
            predict_services = 1
        else:
            predict_services = 2

        if predict_mortality == 'Same as the national average':
            predict_mortality = 1
        elif predict_mortality == 'Above the national average':
            predict_mortality = 2
        elif predict_mortality == 'Below the national average':
            predict_mortality = 3
        else:
            predict_mortality = 4
 
        if predict_care == 'Same as the national average':
            predict_care = 1
        elif predict_care == 'Above the national average':
            predict_care = 2
        elif predict_care == 'Below the national average':
            predict_care = 3
        else:
            predict_care = 4

        if predict_timeliness == 'Same as the national average':
            predict_timeliness = 1
        elif predict_timeliness == 'Above the national average':
            predict_timeliness = 2
        elif predict_timeliness == 'Below the national average':
            predict_timeliness = 3
        else:
            predict_timeliness = 4
        
        if predict_imaging == 'Same as the national average':
            predict_imaging = 1
        elif predict_imaging == 'Above the national average':
            predict_imaging = 2
        elif predict_imaging == 'Below the national average':
            predict_imaging = 3
        else:
            predict_imaging = 4

        input_data = np.array([predict_patient_experience,predict_hospital_type, predict_zip_code, predict_ownership,predict_mortality, predict_care,predict_timeliness,predict_imaging])
        #print("input data is:")
        #print(input_data.reshape(1, -1))
        #print("test data is:")
        #print(test_features[0])
        prediction_result = my_tree.predict(input_data.reshape(1, -1))
        prediction_result_gradient= gradient.predict(input_data.reshape(1, -1))
        #print("prediction is:",prediction_result)
        #prediction = prediction[0][1] # probability of survival
        #data['prediction'] = '{:.1f}% Chance of Survival'.format(prediction * 100)

        return render_template('hospital1.html',predic=prediction_result_gradient,trainacc=training_accuracy,testacc=testing_accuracy)
    return render_template('hospital1.html')


# Data cleanup
df_loan=pd.read_csv("C:/Repositories/loan_status.csv")

df_loan1=df_loan.copy()


df_loan1["Gender"][df_loan1["Gender"]=='Male']=0
df_loan1["Gender"][df_loan1["Gender"]=='Female']=1

df_loan1["Married"][df_loan1["Married"]=='Yes']=0
df_loan1["Married"][df_loan1["Married"]=='No']=1

df_loan1["Education"][df_loan1["Education"]=='Graduate']=0
df_loan1["Education"][df_loan1["Education"]=='Not Graduate']=1

df_loan1["Self_Employed"][df_loan1["Self_Employed"]=='Yes']=0
df_loan1["Self_Employed"][df_loan1["Self_Employed"]=='No']=1

df_loan1["Property_Area"][df_loan1["Property_Area"]=='Urban']=0
df_loan1["Property_Area"][df_loan1["Property_Area"]=='Rural']=1
df_loan1["Property_Area"][df_loan1["Property_Area"]=='Semiurban']=2
#.............
if len(df_loan1.Gender[df_loan1.Gender.isnull() ]) > 0:
    df_loan1.Gender[df_loan1.Gender.isnull() ] = df_loan1.Gender.dropna().mode().values

if len(df_loan1.Married[df_loan1.Married.isnull() ]) > 0:
    df_loan1.Married[df_loan1.Married.isnull() ] = df_loan1.Married.dropna().mode().values
    
if len(df_loan1.Education[df_loan1.Education.isnull() ]) > 0:
    df_loan1.Education[df_loan1.Education.isnull() ] = df_loan1.Education.dropna().mode().values
    
if len(df_loan1.Self_Employed[df_loan1.Self_Employed.isnull() ]) > 0:
    df_loan1.Self_Employed[df_loan1.Self_Employed.isnull() ] = df_loan1.Self_Employed.dropna().mode().values    
    
if len(df_loan1.Property_Area[df_loan1.Property_Area.isnull() ]) > 0:
    df_loan1.Property_Area[df_loan1.Property_Area.isnull() ] = df_loan1.Property_Area.dropna().mode().values
    
if len(df_loan1.Loan_Amount_Term[df_loan1.Loan_Amount_Term.isnull() ]) > 0:
    df_loan1.Loan_Amount_Term[df_loan1.Loan_Amount_Term.isnull() ] = df_loan1.Loan_Amount_Term.dropna().mode().values
 
if len(df_loan1.Credit_History[df_loan1.Credit_History.isnull() ]) > 0:
    df_loan1.Credit_History[df_loan1.Credit_History.isnull() ] = df_loan1.Credit_History.dropna().mode().values               

df_loan1["ApplicantIncome"] = df_loan1["ApplicantIncome"].fillna(df_loan1["ApplicantIncome"].mean())
df_loan1["LoanAmount"] = df_loan1["LoanAmount"].fillna(df_loan1["LoanAmount"].mean()) 
#df_loan1["ApplicantIncome"] = df_loan1["ApplicantIncome"].fillna(df_loan1["ApplicantIncome"].mean())
#df_loan1["Self_Employed"] = df_loan1["Self_Employed"].fillna(df_loan1["LoanAmount"].mean())  
#print (df_loan1["Self_Employed"])
trainingData_loan=df_loan1[["Loan_ID","Loan_Status","Gender","Married","Education","ApplicantIncome","Self_Employed","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area" ]]
from sklearn.model_selection import train_test_split
train_loan,test_loan = train_test_split(trainingData_loan,test_size = 0.2)

target_loan=train_loan["Loan_Status"].values
features_loan=train_loan[["Gender","Married","Education","ApplicantIncome","Self_Employed","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area" ]].values

#my_tree=tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 2,random_state = 1)
#my_tree=my_tree.fit(features,target)
my_tree_loan = RandomForestClassifier(max_depth = 16, min_samples_split=2, n_estimators =100, random_state = 1)
my_tree_loan = my_tree_loan.fit(features_loan,target_loan)
#print(my_tree_loan.feature_importances_)
#print("training accuracy:",my_tree_loan.score(features_loan,target_loan))
training_accuracy_loan = my_tree_loan.score(features_loan,target_loan)

test_features_loan=test_loan[["Gender","Married","Education","ApplicantIncome","Self_Employed","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area" ]].values

my_prediction_loan = my_tree_loan.predict(test_features_loan)
from sklearn.metrics import accuracy_score
#print ("prediction accuracy:",accuracy_score(test_loan["Loan_Status"],my_prediction_loan))
testing_accuracy_loan = accuracy_score(test_loan["Loan_Status"],my_prediction_loan)
@app.route('/loan', methods = ['GET','POST'])
def loan():
    data = {} 
    if request.form:
        # get the form data
        form_data = request.form
        data['form'] = form_data
        predict_gender = form_data['predict_gender']
        predict_married = form_data['predict_married']
        predict_education = form_data['predict_education']
        predict_income = float(form_data['predict_income'])
        predict_employed = form_data['predict_employed']
        predict_amount = float(form_data['predict_amount'])
        predict_term = float(form_data['predict_term'])
        predict_credit = float(form_data['predict_credit'])
        predict_area = form_data['predict_area']
        
        #print(data)
        if predict_gender == 'Male':
            predict_gender = 0
        else:
            predict_gender = 1

        if predict_married == 'Yes':
            predict_married = 0
        else:
            predict_married = 1

        if predict_education == 'Graduate':
            predict_education = 0
        else:
            predict_education = 1

        if predict_employed == 'Yes':
            predict_employed = 0
        else:
            predict_employed = 1

        if predict_area == 'Urban':
            predict_area = 0
        elif predict_area == 'Rural' :
            predict_area= 1
        else:
            predict_area = 2
 
        input_data = np.array([predict_gender, predict_married, predict_education, predict_income, predict_employed, predict_amount, predict_term, predict_credit, predict_area])
        #print("input data is:")
        #print(input_data.reshape(1, -1))
        #print("test data is:")
        #print(test_features_loan[0])
        prediction_result_loan = my_tree_loan.predict(input_data.reshape(1, -1))
        #print("prediction is:",prediction_result_loan)
        #prediction = prediction[0][1] # probability of survival
        #data['prediction'] = '{:.1f}% Chance of Survival'.format(prediction * 100)

        return render_template('loan.html',predic=prediction_result_loan,trainacc=training_accuracy_loan,testacc = testing_accuracy_loan)
    return render_template('loan.html')