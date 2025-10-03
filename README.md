# Previsioni vendite con Temporal Fusion Transformer

Questo progetto fornisce uno script pronto all'uso per addestrare un modello [Temporal Fusion Transformer (TFT)](https://arxiv.org/abs/1912.09363) sui dati di vendita degli ultimi 9 mesi e produrre la previsione dei pezzi necessari (per modello, colore e taglia) per le successive due settimane.

## Formato dei dati di input

Fornisci un file CSV con almeno le seguenti colonne:

| colonna        | descrizione                                                |
|----------------|------------------------------------------------------------|
| `date`         | Data della vendita (formato ISO `YYYY-MM-DD`).             |
| `model`        | Nome o codice del modello.                                 |
| `color`        | Colore.                                                    |
| `size`         | Taglia (stringa o numero).                                 |
| `units_sold`   | Quantità vendute in quella data per la combinazione sopra. |

* Una riga deve rappresentare il totale giornaliero per una combinazione modello/colore/taglia.
* Gli eventuali giorni senza vendite devono essere comunque presenti con `units_sold = 0`. Se mancano, lo script provvede a riempirli automaticamente.

## Installazione delle dipendenze

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Avvio dell'addestramento e delle previsioni

```bash
python -m src.train_tft \
  --data "prova 4" \
  --output previsioni_giornaliere.csv \
  --summary-output fabbisogno_totale.csv
```

> ℹ️ Il parametro `--data` accetta sia un percorso completo al file CSV sia un nome con spazi (es. `"prova 4"` o `"prova 4.csv"`).
> Lo script cercherà automaticamente il file tra la cartella corrente, `data/`, `datasets/` e `Start/`.

Parametri principali:

* `--data`: percorso al CSV con lo storico di 9 mesi.
* `--output`: file CSV con la previsione giornaliera per i 14 giorni successivi per ogni combinazione modello/colore/taglia.
* `--summary-output`: file CSV con il fabbisogno totale nelle 2 settimane per ogni combinazione.
* `--max-encoder-length`: numero di giorni di storico utilizzati dal modello (default 180, viene ridotto automaticamente se lo storico disponibile è inferiore).
* `--max-epochs`: numero massimo di epoche di addestramento (default 60) con early stopping automatico.
* `--batch-size`: dimensione del batch (default 128).

## Output

* **Previsioni giornaliere**: contiene le vendite previste per ciascuna combinazione modello/colore/taglia per ogni giorno delle due settimane successive, con gli intervalli di confidenza P10/P50/P90.
* **Riepilogo due settimane**: aggrega le previsioni giornaliere mostrando il totale previsto nel periodo (sempre con quantili P10/P50/P90), utile per pianificare riassortimenti.

Entrambi i file includono anche l'intervallo di confidenza stimato dal modello.

## Riproducibilità

Lo script imposta automaticamente i semi random per garantire risultati riproducibili e salva il checkpoint del modello migliore nel file `tft_best.ckpt`.

## Note

* Il training sfrutta automaticamente la GPU se disponibile.
* Assicurati che il dataset contenga almeno 9 mesi completi di dati per ogni combinazione modello/colore/taglia.
* Per una maggiore accuratezza puoi arricchire il CSV con covariate addizionali (promozioni, festività, ecc.) aggiungendole come colonne: se sono numeriche verranno gestite automaticamente come variabili note.
