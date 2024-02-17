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

* Regression - Formel der Geraden
* Klassifizierung - Weisst Daten aufgrund von vordefinierten Regeln (bspw. `x < 6`) bestimmte Klassen zu

### Überwachtes Lernen
Die Daten sind von einem Experten entsprechend gelabelt worden (bspw. Diabethiker Ja/Nein).

### Clustering
Findet eine bestimmte anzahl von Gruppen aus einer Menge von Datenpunkten. Das Streumass beschreibt hierbei die Grösse der Cluster. Das Clustering gehört zu dem **unüberwachtem Lernen**.

### Diskriminative Modelle
Modell gibt Entscheidungsgrenze vor, bspw. Support Vektormaschine, Entscheidungsbäume etc.

* Anwendung ist einfach und braucht wenig Rechenzeit


### Generative Modelle
Modell versucht die Daten anhand von Werten wie Durschnitt etc. zu beschreiben (bspw. K-Means, Naive Bayes)

* Auf unbekannten Daten sind Ausreisser leicht erkennbar



