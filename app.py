import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Hasil Ujian", layout="wide")
st.title("📊 Dashboard Analisis Hasil Ujian Siswa")
st.markdown("Analisis Butir Soal & Segmentasi Performa Siswa")

# ==========================================================
# LOAD DATA (UPLOAD USER)
# ==========================================================
uploaded_file = st.file_uploader("/data_simulasi_50_siswa_20_soal.xlsx", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    # Ambil hanya kolom numerik (soal)
    soal = df.select_dtypes(include=['int64', 'float64'])

    # ==========================================================
    # 1️⃣ KPI AKADEMIK
    # ==========================================================
    df["Total_Skor"] = soal.sum(axis=1)
    rata2 = df["Total_Skor"].mean()
    tertinggi = df["Total_Skor"].max()
    terendah = df["Total_Skor"].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Rata-rata Skor", f"{rata2:.2f}")
    col2.metric("🏆 Skor Tertinggi", tertinggi)
    col3.metric("📉 Skor Terendah", terendah)

    st.divider()

    # ==========================================================
    # 2️⃣ DISTRIBUSI & KURVA NORMAL
    # ==========================================================
    st.header("2️⃣ Distribusi Skor")

    fig_dist, ax_dist = plt.subplots(figsize=(6,4))
    ax_dist.hist(df["Total_Skor"], bins=10, density=True)
    mean = df["Total_Skor"].mean()
    std = df["Total_Skor"].std()

    x = np.linspace(df["Total_Skor"].min(), df["Total_Skor"].max(), 100)
    ax_dist.plot(x, norm.pdf(x, mean, std))
    ax_dist.set_title("Distribusi & Kurva Normal Skor")

    st.pyplot(fig_dist)

    st.divider()

    # ==========================================================
    # 3️⃣ TINGKAT KESULITAN
    # ==========================================================
    st.header("3️⃣ Tingkat Kesulitan Soal")

    p_value = soal.mean()

    fig_diff, ax_diff = plt.subplots(figsize=(8,4))
    ax_diff.bar(p_value.index, p_value.values)
    ax_diff.set_ylabel("p-value")
    ax_diff.set_title("Tingkat Kesulitan Soal")
    ax_diff.set_xticklabels(p_value.index, rotation=45)

    st.pyplot(fig_diff)

    st.dataframe(p_value.to_frame("p-value"))

    st.divider()

    # ==========================================================
    # 4️⃣ DAYA BEDA (27%)
    # ==========================================================
    st.header("4️⃣ Daya Beda Soal")

    df_sorted = df.sort_values("Total_Skor", ascending=False)
    n = int(0.27 * len(df))
    atas = df_sorted.head(n)
    bawah = df_sorted.tail(n)

    discrimination = atas[soal.columns].mean() - bawah[soal.columns].mean()

    fig_dis, ax_dis = plt.subplots(figsize=(8,4))
    ax_dis.bar(discrimination.index, discrimination.values)
    ax_dis.set_title("Discrimination Index")
    ax_dis.axhline(0, linestyle="--")

    st.pyplot(fig_dis)
    st.dataframe(discrimination.to_frame("Daya Beda"))

    st.divider()

    # ==========================================================
    # 5️⃣ KORELASI ANTAR SOAL
    # ==========================================================
    st.header("5️⃣ Korelasi Antar Soal")

    corr = soal.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(6,5))
    im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_corr)

    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=90)
    ax_corr.set_yticklabels(corr.columns)

    st.pyplot(fig_corr)

    st.divider()

    # ==========================================================
    # 6️⃣ ANALISIS REGRESI (ITEM → TOTAL)
    # ==========================================================
    st.header("6️⃣ Regresi Linear (Pengaruh Item ke Total Skor)")

    X = sm.add_constant(soal)
    y = df["Total_Skor"]

    model = sm.OLS(y, X).fit()
    coef = model.params[1:]

    fig_reg, ax_reg = plt.subplots(figsize=(8,4))
    ax_reg.bar(coef.index, coef.values)
    ax_reg.set_title("Koefisien Regresi Item")

    st.pyplot(fig_reg)
    st.info(f"📈 R² Model: {model.rsquared:.2f}")

    st.divider()

    # ==========================================================
    # 7️⃣ CLUSTERING SISWA
    # ==========================================================
    st.header("7️⃣ Segmentasi Siswa")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(soal)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster = kmeans.fit_predict(scaled)

    df["Cluster"] = cluster

    cluster_mean = df.groupby("Cluster")["Total_Skor"].mean()

    fig_cluster, ax_cluster = plt.subplots()
    ax_cluster.bar(cluster_mean.index.astype(str), cluster_mean.values)
    ax_cluster.set_title("Rata-rata Skor per Cluster")

    st.pyplot(fig_cluster)

    st.divider()

    # ==========================================================
    # 8️⃣ RADAR PROFIL CLUSTER
    # ==========================================================
    st.header("8️⃣ Radar Profil Cluster")

    cluster_profile = df.groupby("Cluster")[soal.columns].mean()

    labels = soal.columns.tolist()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig_rad = plt.figure(figsize=(6,6))
    ax_rad = plt.subplot(polar=True)

    for i, row in cluster_profile.iterrows():
        values = row.tolist()
        values += values[:1]
        ax_rad.plot(angles, values, label=f"Cluster {i}")
        ax_rad.fill(angles, values, alpha=0.1)

    ax_rad.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax_rad.set_title("Profil Jawaban Cluster")
    ax_rad.legend()

    st.pyplot(fig_rad)

    st.divider()

    # ==========================================================
    # 9️⃣ TAMBAHAN REPRESENTASI: BOXPLOT PERFORMA
    # ==========================================================
    st.header("9️⃣ Boxplot Performa Siswa")

    fig_box, ax_box = plt.subplots()
    ax_box.boxplot(df["Total_Skor"])
    ax_box.set_title("Boxplot Total Skor")

    st.pyplot(fig_box)

    st.success("✅ Dashboard Lengkap ")