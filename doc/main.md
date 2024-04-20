---
titlepage: true
titlepage-color: "ffffff"
titlepage-rule-color: "ffffff"
titlepage-text-color: "000000"
toc-own-page: true
colorlinks: false
title: Zusammenfassung Artificial Intelligence
author:
  - Yannick Hutter
lang: de
date: "17.02.2024"
lof: true
mainfont: Liberation Sans
sansfont: Liberation Sans
monofont: JetBrains Mono
header-left: "\\small \\thetitle"
header-center: "\\small \\leftmark"
header-right: "\\small \\theauthor"
footer-left: "\\leftmark"
footer-center: ""
footer-right: "\\small Seite \\thepage"
...

\newpage

# Unterrichtsnotizen

## 17.02.2024

### Grundsätzliches Vorgehen im Machine Learning

Daten sind durch einen ETL Pipeline entsprechend gecleant und vorverarbeitet worden. Auf Basis dieser Daten soll ein Modell geschaffen werden, welches zukünftige Daten aufgrund der zur Verfügung stehenden Datenmenge vorhersagen kann.

**Beispiele für Modelle**

- Regression - Formel der Geraden
- Klassifizierung - Weisst Daten aufgrund von vordefinierten Regeln (bspw. `x < 6`) bestimmte Klassen zu

### Überwachtes Lernen

Die Daten sind von einem Experten entsprechend gelabelt worden (bspw. Diabethiker Ja/Nein).

### Clustering

Findet eine bestimmte anzahl von Gruppen aus einer Menge von Datenpunkten. Das Streumass beschreibt hierbei die Grösse der Cluster. Das Clustering gehört zu dem **unüberwachtem Lernen**.

### Diskriminative Modelle

Modell gibt Entscheidungsgrenze vor, bspw. Support Vektormaschine, Entscheidungsbäume etc.

- Anwendung ist einfach und braucht wenig Rechenzeit

### Generative Modelle

Modell versucht die Daten anhand von Werten wie Durschnitt etc. zu beschreiben (bspw. K-Means, Naive Bayes)

- Auf unbekannten Daten sind Ausreisser leicht erkennbar

## 23.03.2024

### Sigmoid Funktion

- Der Parameter `b` definiert die **Steilheit des Übergangsbereich (Umschlag)**
- Der Parameter `a` definiert die **Die Stelle des Übergangsbereich (Umschlag) auf der X-Achse**

### Logistische Funktion

- Bei der Logistischen Funktion sollten die Werte **um 0 herum verteilt werden**.
- Die Logistische Funktion ist nur eine **binäre Klassifizierung**, d.h. es können nur zwei Klassen vorausgesagt werden.

#### Mehrere Klassen

Grundsätzlich kann mit der Logistischen Regression nur zwei Klassen vorausgesetzt werden. Es ist jedoch möglich mehrer Klassen vorauszusagen:

- One vs. All - Es gewinnt diejenige Klasse mit dem höchsten Wert. Es werden `K` Modelle benötigt
- One vs. One - Es gewinnt wieder djenige Klasse mit dem höchsten Wert. Bei `K` Modellen werden `K * (K-1) / 2` Modelle benötigt

### Standard-Scaler

- Mittelwert ist 0
- Standardabweichung ist 1
- Entspricht der **Standardnormalverteilung in der Statistik**

## 06.04.2024

### Entscheidungsbäume (Prüfungsrelevant)
Klassifizieren Daten anhand Entscheidungsgrenzen (waagrechte Linien) anhand von Bediengungen. Problematisch ist es, wenn die Daten genau auf der Entscheidungsgrenze liegen, dort ist die Einteilung nicht eindeutig. Entscheidungsbaumgeneratoren sind schlussendlich nur **entropiesenkende Algorithmen**.

**Entropie (Physik) - Mass für Ordnung**
Die Entropie ist das Gegenteil von Ordnung. In der Physik spricht man von Entropie wenn man von einem Wärmeaustausch spricht. Wenn man bspw. zwei Ziegelsteine hat, einen warmen und einen kalten, dann tauschen die Ziegelsteine die Wärme aus. Wenn lange genug gewartet wird, sind die beiden Ziegelsteine gleich warm (mittlere Temperatur zwichen den zwei Ziegelsteinen). In der Regel gielt je schwerer desto mehr Wärmespeicherkapazität ist vorhanden.

> Wichtig: Dieser Prozess ist nicht umkehrbar!

Die Entropie wird genutzt um eine Trennlinie zu finden, wenn dieser Wert **Null erreicht** hat, gibt es eine Trennlinie. Die Trennlinie wird nicht genau bei Null gezogen, es wird ein Sicherheitsabstand einkalkuliert.

> Wichtig: Entscheidungsbaumgeneratoren sind nicht binär, es können mehrere Klassen abgebildet werden

* Mit einem Entscheidungsbaum können nur senkrechte und wagrechte Linien gezeichnet werden
* Schräge Linien können mit Support Vector Machines realisiert werden

### Vorteile Entscheidungsbäume
* Einfaches und schnelles Modell
* Daten müssen kein Scaling durchlaufen
* Viele Klassifikationsprobleme lassen sich hiermit recht gut lösen

### Nachteile Entscheidungsbäume
* Passt sich teils stark den Daten an (Overfitting)
* Modell antwortet nur mit Ja/Nein, keine Prozentangaben

### Konfiguration von Entscheidungsbäume
Entscheidungsbäume können anhand von folgenden Parametern konfiguriert werden:
* Baumtiefe
* Anzahl Datenpunkte aufgrund derer eine Entscheidung getroffen wird

Wenn beispielsweise die `Baumtiefe` beschränkt wird, dann ist unter Umständen die Entscheidungsfindung nicht mehr eindeutig und es kann eine gewisse Unsicherheit bestehen.

## Random Forest
Anstelle eines Baumes werden mehrere Bäume erzeugt, hierbei werden die `Spalten auf mehrere 
verschiedene Bäume verteilt`.

### Vorteile
* Daten müssen nicht aufwändig aufbereitet werden
* Algorithmus kommt mit vielen Spalten klar

### Nachteile
* Modell nicht so anschaulich wie beim einzelnen Baum
* Hoher Rechenaufwand

## Naive Bayes
Ist eine mathematische Formel zur Berechnung von **bedingten Wahrscheinlichkeiten**, d.h. Wahrscheinlichkeiten welche **unter Annahme einer bestimmten Bedingung** eintreten.

### Multinomialer Bayes
Die Warhscheinlichkeit wird mittels Auszählen ermittelt

### Gaussischer Naiver Bayes
Die zwei Werte einer Gaussverteilung ist der Mittelpunkt und der Streufaktor. Es wird geschaut in welcher Gaussische Glockenkurve der Wert liegt und dann die Wahrscheinlichkeiten berechnet.

Der Naive Bayes ist bei elipsenförmigen Datenwolken gut einsetzbar, er hat aber seine Probleme mit einzelnen isolierten Clustern.

### Chancenverhältnis (Odds)
Ist die Wahrscheinlichkeit dass ein Ereignis eintritt zum Verhältnis dass es nicht eintritt.


## Support Vector Machines (SVM)
* Haben im Gegensatz zu Entscheidungsbäumen etc. keine Unsicherheit/Wahrscheinlichkeit (Probability)
* Der Linienverlauf (Kernel) kann linear oder polynomial sein

## Precision vs. Recall
Wer Recall will muss eine Reduzierung der Precision hinnehmen und umgekehrt.

# 13.04.2024

## ROC-Kurve
Ist eine Methode um Modelle zu vergleichen, wenn die Entscheidungsgrenze nicht mittig sondern irgendwo zwischen 1% und 10% gezogen werden würde.

### Golden-Goal
Das Golden Goal ist eine `True Positive Rate (TPR)` von 1.0 und eine `False Positive Rate (FPR)` von 0.0.

### Area under Curve Score (AUC-Score)
Flächeninhalt anhand der ROC-Kurve, der als Indikator für die Beurteilugn von Modellen, bzw. zu deren Optimierung eingesetzt werden kann.

## Hyperparameter Tuning
Kann mithilfe von `Pipelines` realisiert werden. Als Faustregel gilt:
- 60% der Daten zum trainieren
- 20% der Daten zum Hyperparameter optimieren
- 20% der Daten zum testen






