"""
Progetto finale M1 - Analisi di Vendite in una Catena di Negozi
================================================================
Analizza le vendite giornaliere di una catena di negozi di elettronica:
generazione dataset, esplorazione con Pandas, calcoli con NumPy,
visualizzazioni con Matplotlib e analisi avanzata per categoria.
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# ============================================================
# PARTE 1 - DATASET DI BASE
# ============================================================
print("=" * 60)
print("PARTE 1 - Dataset di base")
print("=" * 60)

negozi = ["Milano", "Roma", "Napoli", "Torino", "Bologna"]
prodotti_prezzi = {
    "Smartphone": 499.99,
    "Laptop": 899.99,
    "TV": 649.99,
    "Cuffie": 89.99,
    "Tablet": 349.99,
    "Frigorifero": 599.99,
    "Lavatrice": 449.99,
}
prodotti = list(prodotti_prezzi.keys())

n_righe = 60  # almeno 30 richieste, ne generiamo di più per analisi più solide
righe = []
data_inizio = pd.Timestamp("2023-09-01")

for i in range(n_righe):
    data = data_inizio + pd.Timedelta(days=random.randint(0, 29))
    negozio = random.choice(negozi)
    prodotto = random.choice(prodotti)
    quantita = random.randint(1, 12)
    # piccola variazione realistica sul prezzo di listino
    prezzo_unitario = round(prodotti_prezzi[prodotto] * random.uniform(0.95, 1.05), 2)

    righe.append({
        "Data": data.strftime("%Y-%m-%d"),
        "Negozio": negozio,
        "Prodotto": prodotto,
        "Quantità": quantita,
        "Prezzo_unitario": prezzo_unitario,
    })

df_base = pd.DataFrame(righe).sort_values("Data").reset_index(drop=True)
df_base.to_csv("vendite.csv", index=False)
print(f"File 'vendite.csv' creato con {len(df_base)} righe.")
print(df_base.head())


# ============================================================
# PARTE 2 - IMPORTAZIONE CON PANDAS
# ============================================================
print("\n" + "=" * 60)
print("PARTE 2 - Importazione con Pandas")
print("=" * 60)

df = pd.read_csv("vendite.csv")

print("\nPrime 5 righe (head()):")
print(df.head())

print(f"\nNumero di righe e colonne (shape): {df.shape}")

print("\nInformazioni generali (info()):")
df.info()


# ============================================================
# PARTE 3 - ELABORAZIONI CON PANDAS
# ============================================================
print("\n" + "=" * 60)
print("PARTE 3 - Elaborazioni con Pandas")
print("=" * 60)

df["Incasso"] = df["Quantità"] * df["Prezzo_unitario"]

incasso_totale = df["Incasso"].sum()
incasso_medio_negozio = df.groupby("Negozio")["Incasso"].mean().sort_values(ascending=False)
top3_prodotti_quantita = df.groupby("Prodotto")["Quantità"].sum().sort_values(ascending=False).head(3)
incasso_medio_negozio_prodotto = df.groupby(["Negozio", "Prodotto"])["Incasso"].mean()

print(f"\nIncasso totale della catena: {incasso_totale:.2f} €")

print("\nIncasso medio per negozio:")
print(incasso_medio_negozio.round(2))

print("\nTop 3 prodotti più venduti (per quantità totale):")
print(top3_prodotti_quantita)

print("\nIncasso medio per Negozio e Prodotto:")
print(incasso_medio_negozio_prodotto.round(2))


# ============================================================
# PARTE 4 - USO DI NUMPY
# ============================================================
print("\n" + "=" * 60)
print("PARTE 4 - Uso di NumPy")
print("=" * 60)

q = df["Quantità"].to_numpy()
media = np.mean(q)
minimo = np.min(q)
massimo = np.max(q)
dev_standard = np.std(q)
perc_sopra_media = (q > media).sum() / len(q) * 100

print(f"Media quantità: {media:.2f}")
print(f"Minimo quantità: {minimo}")
print(f"Massimo quantità: {massimo}")
print(f"Deviazione standard: {dev_standard:.2f}")
print(f"Percentuale di vendite sopra la media: {perc_sopra_media:.1f}%")

# Array 2D con Quantità e Prezzo_unitario, calcolo incasso e verifica
arr_2d = df[["Quantità", "Prezzo_unitario"]].to_numpy()
incasso_numpy = arr_2d[:, 0] * arr_2d[:, 1]
incasso_numpy = np.round(incasso_numpy, 2)

verifica_ok = np.allclose(incasso_numpy, df["Incasso"].to_numpy())
print(f"\nArray 2D (Quantità, Prezzo_unitario) - shape: {arr_2d.shape}")
print(f"Incasso calcolato con NumPy corrisponde alla colonna 'Incasso' del DataFrame: {verifica_ok}")


# ============================================================
# PARTE 5 - VISUALIZZAZIONI CON MATPLOTLIB
# ============================================================
print("\n" + "=" * 60)
print("PARTE 5 - Visualizzazioni con Matplotlib")
print("=" * 60)

# Grafico a barre - incasso totale per negozio
plt.figure(figsize=(8, 5))
df.groupby("Negozio")["Incasso"].sum().sort_values(ascending=False).plot(kind="bar", color="steelblue")
plt.title("Incasso totale per negozio")
plt.xlabel("Negozio")
plt.ylabel("Incasso (€)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("grafico_barre_incasso_negozio.png", dpi=150)
plt.close()

# Grafico a torta - percentuale di incassi per prodotto
incasso_per_prodotto = df.groupby("Prodotto")["Incasso"].sum().sort_values(ascending=False)
plt.figure(figsize=(7, 7))
plt.pie(incasso_per_prodotto.values, labels=incasso_per_prodotto.index, autopct="%1.1f%%", startangle=90)
plt.title("Percentuale di incassi per prodotto")
plt.tight_layout()
plt.savefig("grafico_torta_incasso_prodotto.png", dpi=150)
plt.close()

# Grafico a linee - andamento giornaliero degli incassi totali
incasso_giornaliero = df.groupby("Data")["Incasso"].sum().sort_index()
plt.figure(figsize=(9, 5))
plt.plot(incasso_giornaliero.index, incasso_giornaliero.values, marker="o", markersize=3, color="darkorange")
plt.title("Andamento giornaliero degli incassi della catena")
plt.xlabel("Data")
plt.ylabel("Incasso (€)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("grafico_linee_incassi_giornalieri.png", dpi=150)
plt.close()

print("Grafici salvati: grafico_barre_incasso_negozio.png, "
      "grafico_torta_incasso_prodotto.png, grafico_linee_incassi_giornalieri.png")


# ============================================================
# PARTE 6 - ANALISI AVANZATA
# ============================================================
print("\n" + "=" * 60)
print("PARTE 6 - Analisi Avanzata")
print("=" * 60)

categoria_prodotto = {
    "Smartphone": "Informatica",
    "Laptop": "Informatica",
    "Tablet": "Informatica",
    "Cuffie": "Accessori",
    "TV": "Elettrodomestici",
    "Frigorifero": "Elettrodomestici",
    "Lavatrice": "Elettrodomestici",
}

df["Categoria"] = df["Prodotto"].map(categoria_prodotto)

incasso_per_categoria = df.groupby("Categoria")["Incasso"].sum().sort_values(ascending=False)
quantita_media_per_categoria = df.groupby("Categoria")["Quantità"].mean().round(2)

print("Incasso totale per categoria:")
print(incasso_per_categoria.round(2))
print("\nQuantità media venduta per categoria:")
print(quantita_media_per_categoria)

df.to_csv("vendite_analizzate.csv", index=False)
print("\nFile salvato: vendite_analizzate.csv")


# ============================================================
# PARTE 7 - ESTENSIONI
# ============================================================
print("\n" + "=" * 60)
print("PARTE 7 - Estensioni")
print("=" * 60)

# Grafico combinato: incasso medio per categoria (barre) + quantità media (linea)
incasso_medio_categoria = df.groupby("Categoria")["Incasso"].mean().round(2)
quantita_media_categoria = df.groupby("Categoria")["Quantità"].mean().round(2)
categorie_ordinate = incasso_medio_categoria.index

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(categorie_ordinate, incasso_medio_categoria.values, color="mediumseagreen", label="Incasso medio (€)")
ax1.set_xlabel("Categoria")
ax1.set_ylabel("Incasso medio (€)", color="mediumseagreen")
ax1.tick_params(axis="y", labelcolor="mediumseagreen")

ax2 = ax1.twinx()
ax2.plot(categorie_ordinate, quantita_media_categoria.values, color="crimson",
         marker="o", linewidth=2, label="Quantità media")
ax2.set_ylabel("Quantità media venduta", color="crimson")
ax2.tick_params(axis="y", labelcolor="crimson")

plt.title("Incasso medio e quantità media per categoria")
fig.tight_layout()
plt.savefig("grafico_combinato_categoria.png", dpi=150)
plt.close()

print("Grafico combinato salvato: grafico_combinato_categoria.png")


def top_n_prodotti(n=3):
    """Restituisce gli n prodotti più venduti in termini di incasso totale."""
    return df.groupby("Prodotto")["Incasso"].sum().sort_values(ascending=False).head(n)


print("\nTop 3 prodotti per incasso totale:")
print(top_n_prodotti(3).round(2))

print("\n" + "=" * 60)
print("ESECUZIONE COMPLETATA CON SUCCESSO")
print("=" * 60)
