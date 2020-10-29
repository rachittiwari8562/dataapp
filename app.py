import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import math

def main():
	st.title("`------DATA ANALYTICS SOLUTIONS-----`")
	side_bar=st.sidebar
	side_bar.title("Analyze your data")
	side_bar.title("Step -1 Data Preprocessing")
	st.subheader("Upload a File")
	uploaded_file = st.file_uploader("",type="csv")
	@st.cache
	def load_data(uploaded_file):
		if uploaded_file is not None:
			df = pd.read_csv(uploaded_file)
			return df
		else:
			return None
	if uploaded_file:
		df = load_data(uploaded_file)
		st.dataframe(df)
	side_bar.subheader("1.Filtering data")
	col = df.columns
	options = side_bar.multiselect("Choose Attributes which you want to drop",col)
	df = df.drop(columns=options)

	
	side_bar.subheader("2.Performing NAN check")
	na = side_bar.checkbox("Show NaN values")
	nan_num = side_bar.selectbox("Select procedure to handle NAN values for numerical data",["None","mean","median"])
	nan_cat = side_bar.selectbox("Select procedure to handle NAN value for categorial data",["None","most_frequent"])
	if nan_num!="None":
		imputer = SimpleImputer(strategy=nan_num)
		for i in df:
			if df[i].dtype == "float64" or df[i].dtype=="int64":
				k = imputer.fit_transform(df[[i]])
				if k.any():
					df[i]=k
	if nan_cat!="None":
		imputer2 = SimpleImputer(strategy=nan_cat)
		for i in df:
			if df[i].dtype == "object":
				df[i]= imputer2.fit_transform(df[[i]])
	
	def calculate_nan(df):
		na = df.isnull().sum()
		st.header("Nan values in the Dataset are")
		st.write(na)
		if na.sum() >0:
			st.header("Bar plot of the null values")
			st.bar_chart(na)
			null=0
		else:
			st.write("No null values in the dataset")
			null=0
	if na:	
		calculate_nan(df)
	side_bar.subheader("3.Encoding for textual data")
	col = side_bar.multiselect("Choose columns in which it is to be applied",df.columns,key="col")
	if col:
		encoding = side_bar.selectbox("Choose from following techniques",["Label Encoding","One-Hot-key Encoder"])
		if encoding == "Label Encoding":
			labelencoder = LabelEncoder()
			for i in col:
				df[i]=labelencoder.fit_transform(df[i])
		elif encoding=="One-Hot-key Encoder":
			st.write("coming-soon")
	
	side_bar.subheader("4.Outlier Detection and Removal")
	@st.cache
	def outliers(data):
		Q1 = data.quantile(0.25)         #calculating first quantile
		Q3 = data.quantile(0.75)         #calculating third quantile
		IQR = Q3-Q1                      #calculating interquantile range
		lower_whisker = Q1 - 1.5*IQR     #lower-whisker
		upper_whisker = Q3 + 1.5*IQR     #upper-whisker
		outliers=pd.concat((data[data < (Q1-1.5*IQR)], data[data > (Q3+1.5*IQR)]))  #calculating outliers
		return outliers
	
	show_out = side_bar.checkbox("Show Outliers")
	out=[]
	def outlying():
		for i in df:
			if df[i].dtype != "object":
				out.append(outliers(df[i]))
	outlying()
	
	column2 = side_bar.multiselect("Choose Attributes whose outliers are to be removed",[i.name  for i  in out if len(i)>0])
	if column2:
		rep = side_bar.selectbox("Replace all outliers by",["None","Mean","Median"])
		if rep!="None":
			for i in out:
				for j in column2:
					if i.name == j:
						if rep=="Mean":
							df[j][i.index]=df[j].mean()
						else:
							df[j][i.index]=df[j].median()
		out=[]
		outlying()
	if show_out:
		if out:
			st.title("\n\n\n")
			st.header("Outliers Data")
			for j in range(0,math.ceil(len(out)/3)):
				colom = st.beta_columns(3)
				for x in range(3*j,3*(j+1)):
					if x<len(out):
						with colom[x%3]:
							st.write("Total outliers for attribute '"+str(out[x%3+3*j].name)+"' are "+str(len(out[x%3+3*j])))
							st.dataframe(out[x%3+3*j])
				st.title("\n\n\n")

	st.title("\n")
	st.header("Data After Preproccesing")
	st.dataframe(df)

	side_bar.subheader("5.Data description after preproccesing")
	des = pd.DataFrame()
	if side_bar.checkbox("Mean"):
		des["Mean"]=df.mean()
	if side_bar.checkbox("Median"):
		des["Median"]=df.median()
	if side_bar.checkbox("Mode"):
		des["Mode"]=df.mode().transpose()[0]
	if side_bar.checkbox("Standard-Deviation"):
		des["Std"]=df.std()
	if not des.empty:
		st.header("Data Statistic")
		st.write(des)
	if side_bar.checkbox("Boxplot"):
		fig,axes = plt.subplots()
		axes = df.boxplot()
		axes.set_title("Boxplot for various attributes")
		axes.set_ylabel("Data ---->")
		st.pyplot(fig)
if __name__ == '__main__':
	main()