from medguide.index.bm25 import tokenize


def test_keeps_numbers_and_drug_names():
    toks = tokenize("Give Amoxicillin 500 mg/kg for COVID-19; HbA1c >= 6.5%")
    for t in ["amoxicillin", "500", "mg/kg", "covid-19", "hba1c", "6.5"]:
        assert t in toks


def test_drops_stopwords():
    assert tokenize("What is the dose of the drug") == ["dose", "drug"]
