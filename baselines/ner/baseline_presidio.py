from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Configure the NLP engine to use the Portuguese spaCy model
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "pt", "model_name": "pt_core_news_lg"}],
}
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

# Initialize the Analyzer and Anonymizer
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["pt"])
anonymizer = AnonymizerEngine()

# Example text
texto_ata = "No dia 09 de Agosto de 2015, a Sra. Ana Laura Torres, residente na Av. Eva Jesus, chegou na sua Mercedes-Benz com a matrícula 76-41-46."

def main():
    print("BASELINE: MICROSOFT PRESIDIO")

    # nalyze the text to find PII
    resultados = analyzer.analyze(text=texto_ata, language="pt")

    print("Detected entities:")
    for res in resultados:
        print(f" - [{res.entity_type}] (Score: {res.score:.2f}): {texto_ata[res.start:res.end]}")

    # Anonymize (Replace text with the exact entity tag)
    resultado_anonimizado = anonymizer.anonymize(
        text=texto_ata,
        analyzer_results=resultados,
        operators={"DEFAULT": OperatorConfig("replace")}
    )

    print("\nAnonymized Text:")
    print(resultado_anonimizado.text)

if __name__ == "__main__":
    main()