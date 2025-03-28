# Engineer Challenge

Dieses Archiv beinhaltet eine Engineering-Challenge für die HUK-Coburg. Das Ziel ist es, eine offene Aufgabe zu stellen, an der Kandidat:innen ihre Fähigkeiten im Kontext Machine Learning Engineering demonstrieren können. Die tatsächliche Lösung ist daher ebenso interessant, wie der Weg dorthin und die eingesetzten Werkzeuge.

## Aufgabe

Stellen Sie sich vor, Sie arbeiten in einem Team zusammen mit *Data Scientisten* (m/w/d) an einer Lösung für die [Kaggle's Tweet Sentiment Extraction Competition](https://www.kaggle.com/c/tweet-sentiment-extraction). Einer ihrer Kollegen ist Chris Deotte und er schlägt die Lösung im Notebook **model_training.ipynb** vor. Das Team ist sich einig, dass diese Lösung in eine *fiktive Produktionsumgebung* überführt werden soll. Das ist Ihre Aufgabe. Die gewählten Gewichte für das Model finden Sie in **weights_final.h5**.

Stellen Sie das Model im Sinne eines Microservice als Webservice zur Verfügung.
Folgende Punkte sollen der Orientierung dienen:

- Erstellung eines Pythonskriptes für den Webservice
- Kapselung der Model-Inferenz in einem Python-Modul (OOP)
- Erstellung eines Dockerfile zum Hosten der Lösung
- Testkonzept wesentlicher Funktionalitäten
- Bestimmen Sie die Latenz/Request-Zeit des Webservice und schildern Sie konzeptionell wie diese verbessert werden kann

## Hinweise

Das Paket **transformers** stammt von *[Huggingface](https://huggingface.co/)*.
Die Gewichte in **weights_final.h5** wurde mit *Tensorflow 2.7.0* erstellt.
Der Webservice empfängt Text und Sentiment.
Als Antwort gibt es die Wörter zurück, die das Sentiment des Textes ausdrücken.
Der Code ist nicht fehlerfrei.


## Ansprechpartner

[Kevin Hirschmann](mailto:Kevin.Hirschmann@huk-coburg.de)