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
