from src.ui.chat_app import document_title, friendly_citations

DROUGHT = (
    "INTA_CREntreRíos_EEAConcepcióndelUruguay_GorelikZonis_YG_"
    "Impacto_sequía_ganadería_bovina_Gualeguaychu.pdf"
)


def test_known_documents_get_a_readable_title():
    assert document_title(DROUGHT).startswith("Impacto de la sequía")


def test_unknown_documents_fall_back_to_a_cleaned_file_name():
    assert document_title("Informe_nuevo-2026.pdf") == "Informe nuevo 2026"


def test_citations_are_rewritten_for_producers():
    answer = f"La productividad cayó un 26,7% [{DROUGHT}: 14]."
    rewritten = friendly_citations(answer)
    assert ".pdf" not in rewritten
    assert "Impacto de la sequía en la ganadería bovina de Gualeguaychú, pág. 14" in rewritten


def test_fullwidth_brackets_and_abbreviated_names_are_rewritten():
    answer = "Las especies más comunes son R. microplus 【IN…_Guia_identificacion_especies_garrapatas_Entre_Rios.pdf: 7】."
    rewritten = friendly_citations(answer)
    assert "【" not in rewritten and ".pdf" not in rewritten
    assert "Guía para identificar garrapatas de bovinos en Entre Ríos, pág. 7" in rewritten


def test_ambiguous_abbreviations_fall_back_to_a_readable_name():
    rewritten = friendly_citations("Dato [INTA…bovino.pdf: 2]")
    assert ".pdf" not in rewritten
    assert "pág. 2" in rewritten


def test_every_document_in_the_knowledge_base_has_a_title():
    from pathlib import Path

    from src.ui.chat_app import DOCUMENT_TITLES

    raw = {path.name for path in Path("data/raw").glob("*.pdf")}
    assert raw <= set(DOCUMENT_TITLES), f"Missing titles for: {raw - set(DOCUMENT_TITLES)}"
