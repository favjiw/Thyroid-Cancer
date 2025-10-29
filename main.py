# Muhammad Favian Jiwani
# 10123115
# Link Dataset: https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="Preprocessing Data Thyroid Cancer", layout="wide")
st.title("Preprocessing Data Thyroid Cancer")
st.write('Muhammad Favian Jiwani')
st.write('10123115')
st.write('Link Dataset: https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence')

df = pd.read_csv('Thyroid_Diff.csv')
st.success(f"File berhasil dimuat. Jumlah baris: {df.shape[0]}, kolom: {df.shape[1]}")

st.header("1. Info & Tipe Kolom")
st.write("Shape: ", df.shape)
st.write("Kolom: ", list(df.columns))
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)
st.dataframe(df.head())

st.header("2. Missing Values")
missing = df.isnull().sum()
st.write(missing)

st.header("3. Duplikasi Data")
dup_count = df.duplicated().sum()
st.write(f"Jumlah baris duplikat sebelum dihapus: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    st.write("Baris duplikat dihapus.")
st.write(f"Jumlah baris duplikat setelah dihapus: {df.duplicated().sum()}")


st.header("4. Normalisasi Ejaan & Huruf Kecil")
object_cols = df.select_dtypes(include='object').columns
for col in object_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

yn_cols = ['smoking', 'hx smoking', 'hx radiothreapy', 'recurred']
for col in yn_cols:
    if col in df.columns:
        df[col] = df[col].replace({
            'yes': 'yes', 'y': 'yes',
            'no': 'no', 'n': 'no'
        })

if 'gender' in df.columns:
    df['gender'] = df['gender'].replace({'f': 'female', 'm': 'male'})

st.write("Kolom bertipe object setelah dibersihkan:", list(object_cols))
st.dataframe(df.head())

st.header("5. Outlier (IQR) pada kolom numerik penting")
numeric_col = "Age" if "Age" in df.columns else "age"
Q1 = df[numeric_col].quantile(0.25)
Q3 = df[numeric_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df[numeric_col] < lower_bound) | (df[numeric_col] > upper_bound)]

st.write(f"Jumlah outlier pada kolom `{numeric_col}`: {len(outliers)}")
st.write(f"Batas bawah: {lower_bound:.2f}, batas atas: {upper_bound:.2f}")
st.write("Data outlier (jika ada):")
st.dataframe(outliers)

fig, ax = plt.subplots()
sns.scatterplot(x=range(len(df)), y=df[numeric_col], ax=ax, color="skyblue", s=60)
ax.set_title(f"Scatterplot kolom {numeric_col}")
ax.set_xlabel("Index")
ax.set_ylabel(numeric_col)
st.pyplot(fig)

st.header("6. Ketidakseimbangan Kelas (Target)")
target_col = "Recurred" if "Recurred" in df.columns else "recurred"
class_counts = df[target_col].value_counts()
class_percent = df[target_col].value_counts(normalize=True) * 100

st.write("Distribusi kelas:")
st.write(class_counts)
st.write("Persentase (%):")
st.write(class_percent.round(2))

fig2, ax2 = plt.subplots()
sns.countplot(x=df[target_col], palette="pastel", ax=ax2)
ax2.set_title(f"Distribusi Kelas Target ({target_col})")
st.pyplot(fig2)

st.header("Scatterplot Distribusi Umur terhadap Kelas Recurred")
numeric_col = "Age" if "Age" in df.columns else "age"
target_col = "Recurred" if "Recurred" in df.columns else "recurred"

fig, ax = plt.subplots(figsize=(7,4))
sns.scatterplot(
    data=df,
    x=range(len(df)),
    y=numeric_col,
    hue=target_col,
    palette={'no': 'skyblue', 'yes': 'salmon'},
    s=60,
    alpha=0.8
)
ax.set_title(f"Distribusi {numeric_col} terhadap Kelas {target_col}")
ax.set_xlabel("Index Data")
ax.set_ylabel("Age (Usia)")
st.pyplot(fig)

st.header('6. Memeriksa Leakage')
df_encoded = df.copy()
for col in df_encoded.select_dtypes('object'):
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

corr = df_encoded.corr()['Recurred'].sort_values(ascending=False)
st.write(corr)

df = df.drop(columns=['Response', 'Risk'])


st.header("7. Tampilkan 5 Baris Teratas & Terbawah")
col1, col2 = st.columns(2)
with col1:
    st.subheader("5 Baris Teratas")
    st.dataframe(df.head(5))
with col2:
    st.subheader("5 Baris Terbawah")
    st.dataframe(df.tail(5))

