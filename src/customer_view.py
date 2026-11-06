"""The customer's view of the same orders: what happened to each customer, what they
wrote, whether they came back, what date they could have been told, and which contact
could still reach them before they rated the order.

    python src/review_timing.py     # first: story_order.csv and review_timing_*.csv
    python src/customer_view.py     # then: writes report/results/customer_*.csv

Nothing is refitted. The lead-time model is the deployed models/p1a_leadtime.joblib,
scored on the validation and test windows exactly as train_models.py scores them
(the rescored test predictions equal predictions_p1a_test.csv to 1e-9 and the
validation MAE equals train_val_test.csv's 4.7838 - both checked below). Every other
score is read from the saved per-order test predictions.

Definitions (one copy: the report and the dashboard read the CSVs written here)
-------------------------------------------------------------------------------
test window      purchases after config.VAL_END up to config.ANALYSIS_END, i.e. 30 June
                 to 30 August 2018: 12,927 orders placed, 12,855 reviewed
                 (predictions_p2_test.csv), 12,630 delivered (predictions_p1a_test.csv)
promised         order_estimated_delivery_date (always 00:00); the promised day ends 24
                 hours later (eod)
late             delivered after the promised date began - the report's and the models'
                 definition (data_load is_late); late_calendar: after the promised day ended
low              review_score <= config.LOW_REVIEW_MAX_SCORE, on the review data_load keeps
                 (earliest per order). Its text is joined back on all four kept columns
written_before_arrival   delivered, and the review answered before the delivery
overdue_eod      no parcel when the promised day ended: delivered at or after eod, or never
                 delivered and eod inside the extract (last timestamp 2018-10-17)
promise_reach    overdue_eod, and the review not yet written at eod
delivery_reach   delivered, and the review not yet written at delivery
reach_class      'either', 'promised-date message only', 'delivery-day call only', 'neither'
values           R$ per 1,000 REVIEWED test orders (12,855). config.COSTS are assumptions;
                 customer_value_sensitivity.csv varies them, and the efficacy (the share of
                 reached 1-2 star customers a contact turns around) is varied everywhere

Outputs (report/results/)
-------------------------
    customer_outcomes_test.csv       what happened to the customers of the test window, per
                                     1,000 orders placed (or delivered), with the population
                                     named on every row and a Wilson interval
    customer_contact_lists.csv       every contact list: who it reaches, how many a day,
                                     precision, and the reach class of its hits
    customer_value.csv               net R$ per 1,000 reviewed orders by list, by which hits
                                     count, at three efficacies; break-even share and cost
    customer_value_decomposition.csv the slides' R$1,466 / R$775 / R$853 / R$433 split by
                                     whom the contact would have reached
    customer_value_sensitivity.csv   contact cost x review cost x efficacy
    customer_pilot_arms.csv          pilot sizes per contact, on the population each contact
                                     can still reach, and the design on slide 11
    customer_story.csv               the presentation's order, every field the story quotes
    customer_story_events.csv        its timeline, one row per moment (Box 1 and the app)
    customer_journeys_test.csv       one row per reviewed test order: timestamps, ranks,
                                     outcome and reach class (read by the dashboard);
                                     review_predates_order flags the cancelled orders whose
                                     survey is dated before the purchase (a data anomaly)
    customer_lateness_days.csv       1-2 star rate by days early or late, around the
                                     promised day
    customer_complaint_topics.csv    what 1-2 star reviewers wrote about, by whether the
                                     parcel was on time, late or never delivered
    customer_complaint_dictionary.csv  the frozen topic patterns
    customer_complaint_precision.csv how often each topic pattern is right, from a labelled
                                     held-out sample (customer_topic_labels.csv, committed;
                                     AI-assisted labels, see the report's AI Tool Declaration)
    customer_basket.csv              on-time parcels: 1-2 star rate by items / sellers in the
                                     order, and what those reviewers wrote
    customer_survey_timing.csv       late parcels with 1-2 stars: survey and answer timing
                                     against the promised date (is the story order typical?)
    customer_repeat.csv              whether a bad first order stops a customer coming back,
                                     four horizons, two definitions, with tests
    customer_promise_rules.csv       forecast + margin as the promised date, the margin chosen
                                     without test outcomes, against the platform's date
    customer_promise_region.csv      the same, by customer region
    customer_promise_frontier.csv    late rate against mean promise for every margin on a
                                     half-day grid, test and validation (the test columns
                                     are descriptive; choose_buffer() reads only validation)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

import config
import modeling as M
import pilot_design as PD
from data_load import build_orders, load_all
from features import CHECKOUT_FEATURES, build_features

RESULTS = config.PROJECT_ROOT / "report" / "results"
LABELS_FILE = RESULTS / "customer_topic_labels.csv"
# The extract thins out from 22 Aug 2018 (daily orders 250 -> 187, 144, 99 ...), so a
# repeat purchase after that cannot be observed; windows must close by then.
REPEAT_END = pd.Timestamp("2018-08-22")
HORIZONS = (30, 90, 180, 365)

# Reach classes: which contact could still come before the customer rated the order.
REACH_EITHER = "either"
REACH_PROMISE = "promised-date message only"
REACH_DELIVERY = "delivery-day call only"
REACH_NEITHER = "neither"
REACH_CLASSES = (REACH_PROMISE, REACH_DELIVERY, REACH_EITHER, REACH_NEITHER)
REACH_TAG = {REACH_PROMISE: "promise_only", REACH_DELIVERY: "delivery_only",
             REACH_EITHER: "either", REACH_NEITHER: "neither"}

# What happened to the parcel, in the customer's terms.
OUT_ON_TIME = "on time"
OUT_LATE_DAY = "late, on the promised day"
OUT_LATE = "late"
OUT_NEVER = "never delivered"

# Value and pilot settings. Efficacy is the share of reached 1-2 star customers a
# contact turns around; nothing in the data measures it.
EFFICACIES = (1.0, 0.5, 0.25)
SENS_CALL_COST = (2.5, 5.0, 10.0)
SENS_REVIEW_COST = (20.0, 40.0, 80.0)
SENS_EFFICACY = (0.25, 0.5, 1.0)
DELIVERY_DAY_K = (5, 10, 20)
PILOT_TARGET = 0.25          # the smallest relative effect the pilot is sized to detect
PILOT_LISTED_PER_DAY = 10    # contacts listed a day on the slide-11 design and on Arm A

# Days early (-) or late (+) against the promised date, around the promised day.
LATENESS_BINS = [-np.inf, -7, -3, -1, 0, 1, 2, 3, 7, np.inf]
LATENESS_SLUGS = ["early_7plus", "early_3_7", "early_1_3", "early_0_1", "late_0_1",
                  "late_1_2", "late_2_3", "late_3_7", "late_7plus"]
LATENESS_WORDS = ["7 or more days early", "3-7 days early", "1-3 days early",
                  "up to 1 day early", "during the promised day", "1-2 days late",
                  "2-3 days late", "3-7 days late", "more than 7 days late"]

# Half-day grid of customer_promise_frontier.csv (the same expression promise_tables uses).
FRONTIER_GRID = np.round(np.arange(0, 12.01, 0.5), 1)


def require(condition: bool, message: str) -> None:
    """An assertion that survives `python -O`: every reproduction check below uses it."""
    if not condition:
        raise AssertionError(message)


def need(name: str, **kw) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python src/train_models.py`, "
                         "`python src/capacity_metrics.py`, `python src/pilot_design.py` and "
                         "`python src/review_timing.py` first.")
    return pd.read_csv(path, **kw)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


# ===========================================================================
# A. Complaint topics - the dictionary (frozen; validated on a held-out reading)
# ===========================================================================
_INFORMAL = [(r"\bn\b", "nao"), (r"\bq\b", "que"), (r"\bhj\b", "hoje"), (r"\bagr\b", "agora"),
             (r"\bmto?\b", "muito"), (r"\bpq\b", "porque"), (r"\bvcs\b", "voces"), (r"\bvc\b", "voce"),
             (r"\bta\b", "esta"), (r"\bto\b", "estou"), (r"\bmsm\b", "mesmo")]
_INFORMAL = [(re.compile(a), b) for a, b in _INFORMAL]


def informal(s: str) -> str:
    for rx, rep in _INFORMAL:
        s = rx.sub(rep, s)
    return s


# "nao recebi X" where X is not the parcel (a reply, a refund, an invoice, a second item)
_NOT_THE_PARCEL = (r"(?!\s+(?:a\s+|o\s+|as\s+|os\s+|nenhuma?\s+|qualquer\s+|um\s+|uma\s+)?"
                   r"(?:resposta|retorno|reembolso|estorno|dinheiro|valor|e-?mail|contato|nota|nf\b|boleto|"
                   r"codigo|informac|explicac|satisfac|posicao|devoluc|troca|credito|ressarc|ligac|mensage|"
                   r"outr[oa]s?\b|segund[oa]\b|restante|resto|demais|\d+\s*(?:dos|das)\b|um\s+dos|uma\s+das|"
                   r"kit\s+completo|produto\s+(?:certo|correto|completo)|quantidade))")
# "nao chegou o X" where X is a specific part of the order (partial delivery, not the parcel)
_SPECIFIC_PART = r"(?!\s+(?:o|a|os|as|nem|um|uma)\s+(?!produto|pedido|mercadoria|encomenda|compra|minha|meu|ainda|nada|ate)\w)"
_QTY = r"(?:\d+|um|uma|dois|duas|tres|quatro|cinco|seis|dez)"

TOPICS = {
    # 1. The order as a whole never reached the customer, not yet, or could not be fulfilled
    "not_received": (
        r"\bnao\s+(?:recebi|recebemos|recebeu|recebido|recebida|recebir|recebimento)\b" + _NOT_THE_PARCEL +
        r"|\bnao\s+(?:me\s+)?(?:chegou|chegaram)\b" + _SPECIFIC_PART +
        r"|\bnao\s+(?:foi\s+entregue|foram\s+entregues|entregaram|entregou|foi\s+recebid[oa])\b" + _SPECIFIC_PART +
        r"|\bnao\s+entrega\b|\bnao\s+entregue\b|\bnem\s+(?:foi\s+)?(?:entregue|chegou)\b|\bsem\s+receber\b|\bcade\s+(?:o|a|meu|minha)\b"
        r"|\b(?:produto|pedido|mercadoria|encomenda|compra)s?\s+(?:ainda\s+)?nao\s+(?:veio|vieram)\b"
        r"|\b(?:pedido|compra|paguei|comprei)\s+e\s+(?:ainda\s+)?nao\s+(?:veio|chegou)\b"
        r"|\bainda\s+(?:nao\s+)?(?:aguardo|aguarda|aguardando|esperando|espero)\b|\bna\s+espera\b"
        r"|\b(?:nunca|jamais)\s+(?:chegou|chegaram|recebi|foi\s+entregue|veio)"
        r"|\bate\s+(?:o\s+)?(?:agora|hoje|momento|presente(?:\s+momento)?|o\s+momento)[\s,.!]+(?:nada|nao|sem)"
        r"|\b(?:aguardando|aguardo|esperando|espero)\s+(?:a\s+|o\s+|pela?\s+|ate\s+hoje\s+)?(?:entrega|chegada|produto|pedido|mercadoria|encomenda|meu|minha|receber)"
        r"|\bquero\s+(?:o\s+|a\s+)?(?:meu|minha)\s+(?:produto|pedido|mercadoria|encomenda|compra)"
        r"|\bconsta(?:va)?\s+(?:como\s+)?entregue|\bentreg(?:ue|aram|ou)\s+(?:a|para|pra|em)\s+(?:outra\s+pessoa|outro\s+endereco|endereco\s+errado|terceiros)"
        r"|\binforma\w*\s+que\s+(?:o\s+produto\s+|o\s+pedido\s+|a\s+mercadoria\s+)?(?:ja\s+)?(?:foi|tinha\s+sido)\s+entregue"
        r"|\bextraviad\w*|\bextravio\b"
        r"|\b(?:sem|fora\s+de|nao\s+(?:tem|tinha|tinham|havia|temos|tenho)\s+(?:o\s+produto\s+)?(?:mais\s+)?(?:no\s+|em\s+)?)estoque\b"
        r"|\bnao\s+(?:estava|esta|tinha|tem|havia)\s+disponivel|\bindisponivel\b"
        r"|\bnao\s+(?:tinha|tinham|tem|havia|tinhamos)\s+(?:mais\s+)?o\s+produto\b"
    ),
    # 2. Delay / slowness / a missed promised date (delivery, not refunds)
    "late": (
        r"\batras(?:o|os|ad[oa]s?|ou|aram|ar|ando)\b"
        r"|\bdemor(?:a|ou|ad[oa]|ando|ar|aram|ei|am)\b(?!\s+(?:\w+\s+){0,2}(?:estorno|reembolso|devoluc|cancel|respond|resposta|retorno|ressarc))"
        r"|\blent(?:o|a|idao)\b"
        r"|\b(?:fora|passou|passaram|depois|alem|apos|estourou|ultrapass\w*|venc\w*|expirou|excede\w*)\s+(?:d[oa]s?\s+|o\s+|a\s+)?(?:prazo|data\s+(?:prevista|de\s+entrega|estimada|limite|informada|combinada))"
        r"|\bprazo\s+(?:de\s+entrega\s+)?(?:ja\s+)?(?:passou|vencido|vencid[oa]|expirou|expirado|estourado|esgotado|acabou|ultrapassado|excedido|nao\s+(?:foi\s+)?(?:cumprido|respeitado))"
        r"|\bnao\s+(?:cumpr\w*|respeit\w*|honr\w*)\s+(?:o\s+|os\s+|a\s+|as\s+|com\s+)?(?:d[oa]s\s+)?(?:prazo|data)"
        r"|\b(?:prazo|data|previsao)\s+(?:de\s+entrega\s+|prevista\s+|estimada\s+|informada\s+|limite\s+)?(?:era|seria|foi|para|ate|previst[oa]|dia)\b"
        r"|\bprevisao\s+de\s+entrega|\bdata\s+prevista|\bprazo\s+(?:previsto|estipulado|estimado|informado|prometido|dado|combinado)"
        r"|\bera\s+(?:pra|para)\s+(?:ter\s+)?(?:chegad|cheg|entreg|recebid)\w*"
        r"|\bmuito\s+tempo\b|\b(?:mais\s+de|ha|quase)\s+(?:\d+|um|uma|dois|duas|tres|quatro|cinco)\s+(?:dias|semanas?|mes|meses)\b"
        r"|\bchegou\s+(?:com\s+)?(?:\d+\s+dias\s+)?(?:depois|apos|atrasad)"
    ),
    # 3. Wrong item / variant, incomplete or partial delivery
    "wrong_or_missing_item": (
        r"(?<!endereco\s)\berrad[oa]s?\b"
        r"|\b(?:veio|vieram|chegou|chegaram|recebi|mandaram|enviaram|entregaram|enviou|mandou|manda|mandam|envia|enviam)\s+(?:um\s+|uma\s+|o\s+|a\s+)?(?:outr[oa]s?|trocad[oa]s?)\b"
        r"|\b(?:veio|vieram|chegou|chegaram)\s+(?:um\s+|uma\s+|o\s+|a\s+)?diferente\b(?!\s+d[ao]s?\s+(?:foto|imagem|anuncio|anunciad|descri|site|mostrad|propaganda))"
        r"|\boutr[oa]s?\s+(?:produto|cor|modelo|tamanho|sabor|item|marca|tipo|numero|voltagem|estampa)\b|\bdivergente\b"
        r"|\bproduto\s+diferente\s+do\s+(?:que|pedido|comprado|solicitado|encomendado)"
        r"|\bdiferente\s+do\s+que\s+(?:comprei|pedi|solicitei|encomendei|foi\s+(?:pedido|comprado|solicitado))"
        r"|\bnao\s+(?:e|era|foi)\s+o\s+(?:produto|item|modelo|que\s+(?:comprei|pedi|solicitei|encomendei))"
        r"|\bnao\s+(?:enviou|enviaram|mandou|mandaram|veio|vieram|recebi|entregaram)\s+(?:o\s+|a\s+)?(?:produto\s+|item\s+|modelo\s+)?(?:correto|certo|corret[oa]s)\b"
        r"|\bfalt(?:a|am|ava|avam|ou|aram|ando|ad[oa])\b(?!\s+de\s+(?:respeito|atendimento|informac|compromisso|comunicac|consideracao|educac|profissional|seriedade|respons|cuidado|qualidade|retorno|resposta|organizac|transparencia|honestidade|estoque|atencao))(?!\s+(?:confiab|credib|seriedade|respeito|vergonha|atencao|compromisso|profissional))"
        r"|\bincomplet[oa]s?\b|\b(?:veio|vieram|chegou|chegaram)\s+sem\b|\bmetade\b"
        r"|\b(?:so|somente|apenas)\s+(?:me\s+)?(?:recebi|veio|vieram|chegou|chegaram|entregaram|entregou|mandaram|enviaram|um|uma|1|01)\b"
        r"|\b(?:recebi|veio|vieram|chegou|chegaram|entregaram|entregou|enviaram|mandaram)\s+(?:so|somente|apenas)\b"
        r"|\brecebi\s+" + _QTY + r"\w*\s+(?:\w+\s+)?(?:apenas|so|somente)\b"
        r"|\bsendo\s+que\s+(?:eu\s+)?(?:comprei|paguei|pedi)\s+(?:por\s+)?" + _QTY + r"\b"
        r"|\bnao\s+(?:recebi|veio|chegou|vieram|chegaram)\s+(?:o\s+|a\s+|os\s+|as\s+)?(?:outr[oa]s?|segund[oa]|restante|resto|demais|\d+\s+(?:dos|das)|um\s+dos|uma\s+das)"
        r"|\b(?:o|a|os|as)\s+(?:outr[oa]s?|segund[oa]|restante|demais)\s+(?:\w+\s+){0,2}(?:nao|ainda\s+nao)\s+(?:chegou|chegaram|veio|vieram|recebi)"
        r"|\b(?:nao|nem)\s+vei(?:o|eram)\s+(?:o|a|os|as|com|junto|esse|este|essa|esta|nenhum|nenhuma)\b"
        r"|\bnao\s+(?:chegou|chegaram|foi\s+entregue)\s+(?:o|a|os|as|nem|um|uma)\s+(?!produto|pedido|mercadoria|encomenda|compra|minha|meu|ainda|nada|ate)\w"
        r"|\b(?:veio|vieram|chegou|chegaram|entregue|entregues|enviad[oa]s?)\s+(?:\w+\s+)?separad[oa]s?|\bem\s+datas?\s+(?:separadas|diferentes)"
        r"|\bquantidade\s+(?:errada|incorreta|menor|inferior|abaixo|diferente)|\bsimilar\b"
    ),
    # 4. Arrived broken, damaged, dirty/used or not working
    "defective_or_damaged": (
        r"\bdefeit\w*|\bquebr(?:ad[oa]s?|ou|aram)\b|\bdanific\w*|\bavari\w*|\bamass(?:ad[oa]s?|ou)\b"
        r"|\brach(?:ad[oa]s?|ou|adura)\b|\btrinc(?:ad[oa]s?|ou)\b|\bestrag\w*|\brasg(?:ad[oa]s?|ou)\b"
        r"|\barranh\w*|\briscad[oa]s?\b|\bmanchad[oa]s?\b|\bfurad[oa]s?\b|\bvaz(?:ad[oa]s?|ando|ou|amento)\b"
        r"|\bsolt(?:ando|ou|aram)\b|\bdescol\w*|\bdescasc\w*|\bferruge\w*|\benferruj\w*|\bsuj[oa]s?\b"
        r"|\bnao\s+(?:esta\s+|estao\s+)?(?:funciona|funcionou|funcionam|funcionava|funcionando|liga|ligou|carrega|carregou|acende|acendeu|esquenta|conecta|conectou|encaixa|serve|serviu)\b"
        r"|\bnao\s+(?:e\s+)?reconhec\w*|\bparou\s+de\s+funcionar|\bpifou\b|\bqueim(?:ou|ad[oa])\b|\bfunciona\s+mal\b"
        r"|\b(?:embalagem|caixa|pacote)\s+(?:\w+\s+)?(?:violad|abert|rasgad|danificad|amassad|molhad)"
        r"|\bviolad[oa]s?\b|\b(?:produto|veio|chegou|esta|estava|vieram)\s+com\s+problemas?\b|\busad[oa]s?\b"
    ),
    # 5. Quality / not as described / counterfeit / incompatible
    "quality_or_not_as_described": (
        r"(?<!boa\s)(?<!otima\s)(?<!excelente\s)(?<!alta\s)(?<!bem\s)(?<!com\s)\bqualidade\b(?!\s*(?:,|\.|e\s+(?:boa|otima|excelente))\s*(?:so|mas|porem|apenas)\b)(?!\s+(?:e\s+|eh\s+)?(?:boa|otima|excelente|razoavel|ok|satisfatoria))"
        r"|\bmaterial\s+(?:\w+[\s.]+)?(?:ruim|fraco|fino|vagabundo|inferior|horrivel|pessimo|de\s+(?:pessima|baixa|ma))"
        r"|\bacabamento\s+(?:\w+\s+)?(?:ruim|pessimo|horrivel|mal|grosseiro)|\bfragilidade\b|\bbem\s+simples\b"
        r"|\bdiferente\s+d[ao]s?\s+(?:foto|imagem|anuncio|anunciad[oa]|descricao|descrit[oa]|site|mostrad[oa]|propaganda|ofertad[oa]|apresentad[oa]|exposto)"
        r"|\bnao\s+(?:e|era|corresponde|condiz|confere|bate)\s+(?:com\s+)?(?:[oa]\s+|ao\s+|a\s+)?(?:foto|imagem|anuncio|anunciad[oa]|descricao|descrit[oa]|site|propaganda|especificac)"
        r"|\bnao\s+(?:e|era)\s+(?:o\s+)?(?:que|como)\s+(?:eu\s+)?(?:esperava|imaginava|pensava|estava\s+(?:no\s+)?(?:anunciado|exposto|mostrado)|aparece|aparecia|mostrava)"
        r"|\bpropaganda\s+enganosa|\bfalsificad\w*|\breplica\b|\bpirata\b|\boriginal\b"
        r"|\b(?:e|eh|era|sao|parece(?:\s+ser)?|produto|item|mercadoria|cartao|relogio|perfume)\s+fals[oa]s?\b"
        r"|(?<!caixa\s)(?<!etiqueta\s)(?<!como\s)\bfragil\b|\bvagabund\w*|\bmuito\s+(?:pequen[oa]|fin[oa]|frac[oa]|simples|inferior|fraquinh[oa])"
        r"|\bmenor\s+(?:que|do\s+que|doque)\b|\btamanho\s+(?:menor|diferente|pequeno|errado)"
        r"|\b(?:diferente|nao\s+(?:e|era|corresponde|condiz|confere|bate|veio)|fora|ao\s+contrario)\b[^.!?]{0,25}\banunci\w*"
        r"|\banunci\w*\s+(?:diz|dizia|mostra|mostrava|informa|informava|falava|fala|consta|constava|dava)\b|\b(?:como|conforme)\s+(?:o\s+|no\s+)?anunci\w*(?!\s*,)"
        r"|\b(?:diferente|igual|como|conforme|parecid[oa]|mostra\w*|na|da|das|nas|pela)\s+(?:a\s+|as\s+|o\s+)?(?:fotos?|imagem|imagens)\b"
        r"|\bespecificac\w*|\bdescricao\b|\b(?:in)?compativel\b"
    ),
    # 6. Seller / platform service, information and communication
    "seller_service_or_communication": (
        r"\batendimento\b|\bsac\b|\bdescaso\b|\bdesrespeito\b|\bdescompromisso\b"
        r"|\bsem\s+(?:nenhuma\s+|qualquer\s+)?(?:resposta|retorno|solucao|resolucao|explicacao|satisfacao|informac\w*|posicionamento)"
        r"|\b(?:nenhuma|nenhum|qualquer)\s+(?:resposta|retorno|satisfacao|informacao|explicacao|posicao|posicionamento|solucao|contato)"
        r"|\bnao\s+(?:me\s+|nos\s+)?(?:respond\w*|retorn\w*|resolv\w*|atend\w*|deram\s+(?:retorno|resposta|satisfacao)|da[om]?\s+(?:retorno|resposta|satisfacao))"
        r"|\bnao\s+(?:obtive|tive|recebi|recebo|tenho)\s+(?:\w+\s+){0,4}(?:resposta|retorno|satisfacao|solucao|informac\w*)"
        r"|\bfalta\s+de\s+(?:respeito|atendimento|informac\w*|compromisso|comunicac\w*|consideracao|educac\w*|profissional\w*|seriedade|respons\w*|retorno|resposta|transparencia|organizac\w*)"
        r"|\bninguem\s+(?:responde|resolve|atende|da|me|retorna|deu|entra)|\bnao\s+(?:consigo|consegui|conseguimos)\s+(?:falar|contato|contatar|resolver|entrar)"
        r"|\b(?:entrei|entramos|tentei|tento|tentamos)\s+(?:\w+\s+){0,2}contato|\b(?:fiz|faco|fizemos)\s+contato|\bcontatei\b"
        r"|\b(?:aguard|esper)\w*\s+(?:um\s+|uma\s+|o\s+|a\s+)?(?:retorno|resposta|esclarecimento|posicionamento|contato|solucao|posicao)"
        r"|\bdar\s+satisfacao|\b(?:liguei|ligacoes|whatsapp|ouvidoria|reclame\s+aqui|procon|protocolo)\b"
        r"|\brastreamento|\brastreio\b|\brastrear\b|\bconfus\w*|\bnao\s+(?:fui|fomos)\s+(?:avisad|informad|comunicad|notificad)\w*"
    ),
    # 7. Refund, cancellation, return or exchange
    "refund_cancel_return": (
        r"\bdevoluc\w*|\bdevolv\w*|\breembols\w*|\bestorn\w*|\bressarc\w*|\bcancel\w*"
        r"|\bdinheiro\s+(?:de\s+)?volta|\bmeu\s+dinheiro\b|\bdinheiro\s+(?:devolvido|reembolsado|estornado)"
        r"|\btroca\b|\btrocar\b"
    ),
}
ORDER = list(TOPICS)
LABELS = {
    "not_received": "not received (yet) / not fulfilled",
    "late": "late / slow / missed date",
    "wrong_or_missing_item": "wrong or missing item",
    "defective_or_damaged": "defective or damaged",
    "quality_or_not_as_described": "quality / not as described",
    "seller_service_or_communication": "seller service / communication",
    "refund_cancel_return": "refund / cancel / return",
}
COMPILED = {k: re.compile(v) for k, v in TOPICS.items()}


def normalise(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\r\n\t]+", " ", s)
    return informal(re.sub(r"\s+", " ", s).strip())


def review_text(df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """The text of the SAME review data_load keeps per order (earliest creation date).

    data_load._reviews_per_order uses groupby().first(), which takes the first
    non-null value column by column - harmless for the score and timestamps, which
    are never null, but it would splice a message from one review onto the score of
    another. So the text is joined back on all four kept columns instead.
    """
    key = ["order_id", "review_score", "review_creation_date", "review_answer_timestamp"]
    out = df.dropna(subset=["review_score"]).merge(
        reviews[key + ["review_id", "review_comment_title", "review_comment_message"]],
        on=key, how="left").drop_duplicates("order_id", keep="first")
    out["txt"] = (out["review_comment_title"].map(normalise) + " | "
                  + out["review_comment_message"].map(normalise)).str.strip(" |")
    out["has_text"] = out["txt"].str.len() > 0
    out["outcome"] = np.select(
        [out["order_delivered_customer_date"].isna(), out["is_late"] == 1, out["is_late"] == 0],
        ["never delivered", "late", "on time"], "other")
    for k, rx in COMPILED.items():
        out[k] = out["txt"].str.contains(rx)
    out["any_topic"] = out[ORDER].any(axis=1)
    return out


def complaint_topics(rv: pd.DataFrame) -> pd.DataFrame:
    low = rv[(rv["review_score"] <= config.LOW_REVIEW_MAX_SCORE)]
    rows = []
    for name, g in [("all", low)] + [(o, low[low["outcome"] == o])
                                     for o in ("on time", "late", "never delivered")]:
        t = g[g["has_text"]]
        r = {"group": name, "low_reviews": len(g), "with_text": len(t),
             "with_text_pct": 100 * len(t) / len(g)}
        r.update({f"{k}_pct": 100 * t[k].mean() for k in ORDER})
        r["no_topic_pct"] = 100 * (~t["any_topic"]).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def complaint_precision(rv: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Precision of each pattern on the labelled held-out sample (seed 3, 25 per topic)."""
    low = rv[(rv["review_score"] <= config.LOW_REVIEW_MAX_SCORE) & rv["has_text"]]
    rows = []
    for topic in ORDER:
        lab = labels[labels["topic"] == topic]
        # the labelled comments must still be matched by the pattern as frozen
        still = low.set_index("order_id").loc[lab["order_id"], topic]
        require(bool(still.all()), f"{topic}: labelled comments no longer match - the dictionary changed")
        k, n = int(lab["correct"].sum()), len(lab)
        lo, hi = wilson(k, n)
        rows.append({"topic": topic, "label": LABELS[topic], "matched": int(low[topic].sum()),
                     "read": n, "correct": k, "precision_pct": 100 * k / n,
                     "ci95_lo_pct": 100 * lo, "ci95_hi_pct": 100 * hi})
    return pd.DataFrame(rows)


def complaint_dictionary() -> pd.DataFrame:
    """The frozen patterns, one row per topic, applied to accent-stripped, lower-cased
    title + message with common abbreviations expanded (normalise / informal)."""
    return pd.DataFrame([{"topic": k, "label": LABELS[k], "pattern": TOPICS[k]} for k in ORDER])


def basket_table(rv: pd.DataFrame) -> pd.DataFrame:
    """On-time parcels only: 1-2 star rate by how many items / sellers the order had, and
    how many of those reviewers say part of the order was wrong, missing or not received.
    The order carries ONE delivery timestamp, so 'delivered on time' does not mean that
    every item arrived."""
    ot = rv[rv["outcome"] == "on time"].copy()
    ot["low"] = ot["review_score"] <= config.LOW_REVIEW_MAX_SCORE
    ot["part_missing"] = ot["wrong_or_missing_item"] | ot["not_received"]
    groups = [("1 item, 1 seller", (ot["n_items"] == 1) & (ot["n_sellers"] == 1)),
              ("2+ items, 1 seller", (ot["n_items"] > 1) & (ot["n_sellers"] == 1)),
              ("2+ sellers", ot["n_sellers"] > 1),
              ("all on-time", pd.Series(True, index=ot.index))]
    rows = []
    for name, m in groups:
        g = ot[m]
        k, n = int(g["low"].sum()), len(g)
        lo, hi = wilson(k, n)
        c = g[g["low"] & g["has_text"]]
        rows.append({"basket": name, "orders": n, "share_of_on_time_orders_pct": 100 * n / len(ot),
                     "low_rate_pct": 100 * k / n, "ci95_lo_pct": 100 * lo, "ci95_hi_pct": 100 * hi,
                     "share_of_on_time_low_reviews_pct": 100 * k / int(ot["low"].sum()),
                     "commented_low": len(c),
                     "says_wrong_missing_or_not_received_pct": 100 * c["part_missing"].mean(),
                     "says_defective_pct": 100 * c["defective_or_damaged"].mean(),
                     "says_quality_pct": 100 * c["quality_or_not_as_described"].mean()})
    return pd.DataFrame(rows)


def survey_timing(df: pd.DataFrame) -> pd.DataFrame:
    """How typical is the story order? Late parcels with a 1-2 star review: when was the
    survey sent and answered, measured from the promised date (00:00), and was it written
    before the parcel arrived. Analysis window and test window."""
    rows = []
    for name, f in [("analysis window", df), ("test window", df[df["order_purchase_timestamp"] > config.VAL_END])]:
        ll = f[(f["is_late"] == 1) & (f["review_score"] <= config.LOW_REVIEW_MAX_SCORE)]
        prom = ll["order_estimated_delivery_date"]
        sent = (ll["review_creation_date"] - prom).dt.days
        ans = (ll["review_answer_timestamp"] - prom).dt.total_seconds() / 86400
        rows.append({"window": name, "late_low_reviews": len(ll),
                     "survey_days_after_promise_median": sent.median(),
                     "survey_exactly_2_days_after_pct": 100 * (sent == 2).mean(),
                     "survey_on_or_after_promise_pct": 100 * (sent >= 0).mean(),
                     "answer_days_after_promise_median": ans.median(),
                     "answered_on_or_after_promise_pct": 100 * (ans >= 0).mean(),
                     "answered_within_1_day_of_promise_pct": 100 * ans.between(0, 1, inclusive="left").mean(),
                     "written_before_arrival_pct": 100 * (ll["review_answer_timestamp"]
                                                          < ll["order_delivered_customer_date"]).mean()})
    return pd.DataFrame(rows)


def lateness_days(df: pd.DataFrame) -> pd.DataFrame:
    """1-2 star rate of delivered, reviewed analysis-window orders by how many days
    before or after the promised date (00:00) the parcel arrived. The (0, 1] bucket is a
    delivery during the promised day itself, which the report's definition counts as late."""
    d = df.dropna(subset=["delivery_days", "review_score"])
    low = d["review_score"] <= config.LOW_REVIEW_MAX_SCORE
    b = pd.cut(d["delay_vs_estimate_days"], bins=LATENESS_BINS, right=True)
    rows = []
    for interval, slug, words in zip(b.cat.categories, LATENESS_SLUGS, LATENESS_WORDS):
        m = b == interval
        k, n = int(low[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        rows.append({"bucket": f"({interval.left:g}, {interval.right:g}]".replace("inf]", "inf)"),
                     "slug": slug, "label": words, "n": n, "low": k, "low_pct": 100 * k / n,
                     "ci95_lo_pct": 100 * lo, "ci95_hi_pct": 100 * hi})
    out = pd.DataFrame(rows)
    require(int(out["n"].sum()) == len(d), "lateness buckets must cover every delivered, reviewed order")
    return out


# ===========================================================================
# B. Repeat purchase by first-order experience
# ===========================================================================
def first_orders(o: pd.DataFrame) -> pd.DataFrame:
    """Each customer_unique_id's first order anywhere in the extract, with the time of
    their next order: any later order on a later calendar day (V1), and any order placed
    after the first one was delivered (V2). Same-day orders are excluded because 29% of
    'repeat' orders follow the previous one on the same day - split baskets, not returns."""
    o = o.sort_values(["customer_unique_id", "order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    o["k"] = o.groupby("customer_unique_id").cumcount()
    f = o[o["k"] == 0].set_index("customer_unique_id").copy()
    later = o[o["k"] > 0][["customer_unique_id", "order_purchase_timestamp"]].join(
        f[["order_purchase_timestamp", "order_delivered_customer_date"]].rename(
            columns={"order_purchase_timestamp": "t1", "order_delivered_customer_date": "d1"}),
        on="customer_unique_id")
    nxt = lambda m: later[m].groupby("customer_unique_id")["order_purchase_timestamp"].min()
    f["next_later_day"] = nxt(later["order_purchase_timestamp"].dt.normalize() > later["t1"].dt.normalize())
    f["next_after_delivery"] = nxt(later["order_purchase_timestamp"] > later["d1"])
    f["experience"] = np.select([f["order_delivered_customer_date"].isna(), f["is_late"] == 1, f["is_late"] == 0],
                                ["never delivered", "late", "on time"], "other")
    f["first_review"] = pd.cut(f["review_score"], [0, 2, 3, 5], labels=["1-2 star", "3 star", "4-5 star"]).astype(str)
    f.loc[f["review_score"].isna(), "first_review"] = "no review"
    f["quarter"] = f["order_purchase_timestamp"].dt.to_period("Q").astype(str)
    return f[f["order_purchase_timestamp"] >= config.ANALYSIS_START]


def _compare(f: pd.DataFrame, col: str, a, b) -> dict:
    ga, gb = f.loc[f[col] == a, "rep"], f.loc[f[col] == b, "rep"]
    n1, n2, k1, k2 = len(ga), len(gb), int(ga.sum()), int(gb.sum())
    p1, p2, pp = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se0 = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    # Mantel-Haenszel odds ratio across customer region x purchase quarter strata, with the
    # Robins-Breslow-Greenland interval and the Cochran-Mantel-Haenszel test, in numpy/scipy
    # so the script adds no dependency (checked equal to the statsmodels implementation).
    tabs = []
    for _, s_ in f[f[col].isin([a, b])].groupby([f["customer_region"].fillna("NA"), "quarter"]):
        t = np.array([[((s_[col] == a) & s_["rep"]).sum(), ((s_[col] == a) & ~s_["rep"]).sum()],
                      [((s_[col] == b) & s_["rep"]).sum(), ((s_[col] == b) & ~s_["rep"]).sum()]], float)
        if (t.sum(axis=1) > 0).all() and t[:, 0].sum() > 0:
            tabs.append(t)
    T = np.array(tabs)
    A_, B_, C_, D_ = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    N_ = T.sum(axis=(1, 2))
    R_, S_ = A_ * D_ / N_, B_ * C_ / N_
    P_, Q_ = (A_ + D_) / N_, (B_ + C_) / N_
    mh_or = R_.sum() / S_.sum()
    var_log = ((P_ * R_).sum() / (2 * R_.sum() ** 2)
               + (P_ * S_ + Q_ * R_).sum() / (2 * R_.sum() * S_.sum())
               + (Q_ * S_).sum() / (2 * S_.sum() ** 2))
    half = norm.ppf(0.975) * np.sqrt(var_log)
    mh_lo, mh_hi = mh_or * np.exp(-half), mh_or * np.exp(half)
    E_ = (A_ + B_) * (A_ + C_) / N_
    V_ = (A_ + B_) * (C_ + D_) * (A_ + C_) * (B_ + D_) / (N_ ** 2 * (N_ - 1))
    keep = N_ > 1
    cmh = (A_[keep].sum() - E_[keep].sum()) ** 2 / V_[keep].sum()
    return {"group": str(a), "reference": str(b), "n": n1, "repeat_pct": 100 * p1,
            "ci95_lo_pct": 100 * l1, "ci95_hi_pct": 100 * u1, "ref_n": n2, "ref_repeat_pct": 100 * p2,
            "diff_pp": 100 * d,
            "diff_ci95_lo_pp": 100 * (d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),   # Newcombe
            "diff_ci95_hi_pp": 100 * (d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)),
            "p_value": 2 * norm.sf(abs(d / se0)),
            "mde80_pp": 100 * (norm.ppf(0.975) + norm.ppf(0.80)) * se0,
            "mh_odds_ratio": mh_or, "mh_ci95_lo": mh_lo, "mh_ci95_hi": mh_hi,
            "mh_p_value": chi2.sf(cmh, 1)}


def repeat_table(first: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version, anchor, nxt in [("V1 later calendar day, from purchase", "order_purchase_timestamp", "next_later_day"),
                                 ("V2 after delivery, from delivery", "order_delivered_customer_date", "next_after_delivery")]:
        for h in HORIZONS:
            f = first[first[anchor].notna() & (first[anchor] + pd.Timedelta(days=h) <= REPEAT_END)].copy()
            f["rep"] = ((f[nxt] - f[anchor]) <= pd.Timedelta(days=h)).fillna(False).astype(bool)
            for col, a, b in [("experience", "late", "on time"), ("first_review", "1-2 star", "4-5 star"),
                              ("first_review", "3 star", "4-5 star"), ("experience", "never delivered", "on time")]:
                if version.startswith("V2") and a == "never delivered":
                    continue
                r = _compare(f, col, a, b)
                r.update({"version": version, "horizon_days": h, "cohort_n": len(f),
                          "cohort_repeat_pct": 100 * f["rep"].mean(),
                          "cohort_first": str(f["order_purchase_timestamp"].min().date()),
                          "cohort_last": str(f["order_purchase_timestamp"].max().date())})
                rows.append(r)
    return pd.DataFrame(rows)


# ===========================================================================
# C. A promise the customer can rely on: forecast + buffer
# ===========================================================================
GRID = np.round(np.arange(-3.0, 25.0001, 0.1), 1)


def score_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """The deployed regressor on validation and test - no refit, asserted identical."""
    est = M.load_model(config.MODELS_DIR / "p1a_leadtime.joblib", strict=True)["estimator"]
    ts = df["order_purchase_timestamp"]
    parts = []
    for name, m in [("validation", (ts > config.TRAIN_END) & (ts <= config.VAL_END)), ("test", ts > config.VAL_END)]:
        p = df[m].dropna(subset=["target_delivery_days"]).copy()
        p["pred_days"] = est.predict(p[CHECKOUT_FEATURES])
        p["period"] = name
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    ref = need("predictions_p1a_test.csv")
    chk = out[out["period"] == "test"].merge(ref[["order_id", "pred_days"]], on="order_id", suffixes=("", "_ref"))
    require(len(chk) == len(ref) and bool(np.allclose(chk["pred_days"], chk["pred_days_ref"], atol=1e-9)),
            "rescored test forecasts differ from predictions_p1a_test.csv - the model or the features changed")
    return out


def promise_days(f: pd.DataFrame, b: float, whole_day: bool = True) -> pd.Series:
    """Promised days for forecast + b. whole_day rounds the moment UP to the next midnight,
    i.e. a calendar date like Olist's order_estimated_delivery_date (always 00:00)."""
    t = f["order_purchase_timestamp"] + pd.to_timedelta(f["pred_days"] + b, unit="D")
    if whole_day:
        t = t.dt.ceil("D")
    return (t - f["order_purchase_timestamp"]).dt.total_seconds() / 86400


def _promise_row(f: pd.DataFrame, prom: pd.Series) -> dict:
    late = f["delivery_days"] > prom                       # the repository's is_late, any promise
    late_cd = f["delivery_days"] >= prom + 1               # delivered after the END of the promised day
    gap = prom - f["delivery_days"]
    return {"n": len(f), "late_pct": 100 * late.mean(), "late_calendar_day_pct": 100 * late_cd.mean(),
            "mean_promised_days": prom.mean(), "median_promised_days": prom.median(),
            "mean_actual_days": f["delivery_days"].mean(), "mean_told_minus_waited_days": gap.mean(),
            "mean_days_late_if_late": (-gap[late]).mean(),
            "pct_arrived_7plus_days_early": 100 * (gap >= 7).mean()}


def promise_tables(pred: pd.DataFrame):
    val, test = pred[pred["period"] == "validation"], pred[pred["period"] == "test"].copy()
    olist_val_late = (val["delivery_days"] > val["estimated_days"]).mean()
    lates = np.array([(val["delivery_days"] > promise_days(val, b)).mean() for b in GRID])
    require(bool((lates <= olist_val_late).any()), "no margin on the grid matches the platform's validation late rate")
    b_r1 = float(GRID[np.argmax(lates <= olist_val_late)])                 # needs val outcomes only
    means_v = np.array([promise_days(val, b).mean() for b in GRID])
    b_r2v = float(GRID[np.argmin(np.abs(means_v - val["estimated_days"].mean()))])
    means_t = np.array([promise_days(test, b).mean() for b in GRID])
    b_r2 = float(GRID[np.argmin(np.abs(means_t - test["estimated_days"].mean()))])   # label-free
    rules = [("Olist's own promise", None, "", lambda f: f["estimated_days"]),
             ("Forecast alone (no buffer)", 0.0, "", lambda f: f["pred_days"]),
             ("R1 buffer: validation late rate <= Olist's", b_r1, "validation outcomes", lambda f: promise_days(f, b_r1)),
             ("R2 buffer: same mean promise as Olist (label-free)", b_r2, "test promises and forecasts only", lambda f: promise_days(f, b_r2)),
             ("R2-val buffer: validation mean promise = Olist's", b_r2v, "validation promises", lambda f: promise_days(f, b_r2v))]
    rows = []
    for name, b, chosen_on, fn in rules:
        for per, f in (("validation", val), ("test", test)):
            r = {"period": per, "rule": name, "buffer_days": b, "chosen_on": chosen_on}
            r.update(_promise_row(f, fn(f)))
            rows.append(r)
    test["region_group"] = np.select(
        [test["customer_region"].isin(["North", "Northeast"]), test["customer_region"].eq("Southeast")],
        ["North + Northeast", "Southeast"], "South + Central-West")
    reg = []
    for grp, g in list(test.groupby("region_group")) + [("All", test)]:
        for name, b, _, fn in rules[:4]:
            prom = fn(g)
            r = {"region_group": grp, "rule": name, "buffer_days": b}
            r.update(_promise_row(g, prom))
            if b is not None and name.startswith("R"):
                d = prom - g["estimated_days"]
                r.update({"pct_earlier_date_than_olist": 100 * (d <= -1).mean(),
                          "pct_later_date_than_olist": 100 * (d >= 1).mean(),
                          "mean_change_vs_olist_days": d.mean()})
            reg.append(r)
    frontier = pd.DataFrame([{"buffer_days": b, **{k: v for k, v in _promise_row(test, promise_days(test, b)).items()
                                                   if k in ("late_pct", "late_calendar_day_pct", "mean_promised_days")}}
                             for b in np.round(np.arange(0, 12.01, 0.5), 1)])
    olist_point = _promise_row(test, test["estimated_days"])
    frontier = pd.concat([frontier, pd.DataFrame([{"buffer_days": np.nan, "late_pct": olist_point["late_pct"],
                                                   "late_calendar_day_pct": olist_point["late_calendar_day_pct"],
                                                   "mean_promised_days": olist_point["mean_promised_days"]}])],
                         ignore_index=True)
    return pd.DataFrame(rows), pd.DataFrame(reg), frontier


def frontier_validation(pred: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    """Add the validation late rate and mean promise for every margin of the frontier
    (and for the platform's own date on its row without a margin), so the margin choice
    can be re-run from this file alone: choose_buffer() reads only these columns."""
    val = pred[pred["period"] == "validation"]
    out = frontier.copy()
    require(np.array_equal(out["buffer_days"].dropna().to_numpy(), FRONTIER_GRID),
            "the frontier grid changed")
    late, mean = [], []
    for b in out["buffer_days"]:
        prom = val["estimated_days"] if pd.isna(b) else promise_days(val, float(b))
        late.append(100 * (val["delivery_days"] > prom).mean())
        mean.append(prom.mean())
    out["val_late_pct"] = late
    out["val_mean_promised_days"] = mean
    return out


def choose_buffer(frontier: pd.DataFrame, olist_val_late_pct: float | None = None) -> float:
    """Rule R1: the smallest margin on the grid whose VALIDATION late rate is no higher
    than the platform's own validation late rate. Reads only buffer_days and val_late_pct
    (and, if no rate is passed, the platform's row - the one without a margin), so the
    test-window columns cannot influence it."""
    if olist_val_late_pct is None:
        olist_val_late_pct = float(frontier.loc[frontier["buffer_days"].isna(), "val_late_pct"].iloc[0])
    f = frontier.dropna(subset=["buffer_days"]).sort_values("buffer_days", kind="stable")
    ok = f[f["val_late_pct"] <= olist_val_late_pct]
    if ok.empty:
        raise ValueError("no margin on the grid matches the platform's validation late rate")
    return float(ok["buffer_days"].iloc[0])


def requote(purchase: pd.Series, pred_days: pd.Series, buffer_days: float) -> pd.Series:
    """The date the forecast plus the margin would quote: purchase + forecast + margin,
    rounded UP to the next midnight (a calendar date, like the platform's promise)."""
    return (purchase + pd.to_timedelta(pred_days + buffer_days, unit="D")).dt.ceil("D")


# ===========================================================================
# D. The customers of the test window: what happened, and who could still be reached
# ===========================================================================
def customer_flags(t: pd.DataFrame, last_ts: pd.Timestamp) -> pd.DataFrame:
    """Every definition of the module docstring, one row per order of `t`."""
    out = t.copy()
    dl, ans = out["order_delivered_customer_date"], out["review_answer_timestamp"]
    out["promised"] = out["order_estimated_delivery_date"]
    out["eod"] = out["promised"] + pd.Timedelta(days=1)
    out["delivered"] = dl.notna()
    out["never"] = ~out["delivered"]
    out["late"] = out["delivered"] & (dl > out["promised"])
    out["late_calendar"] = out["delivered"] & (dl >= out["eod"])
    out["reviewed"] = out["review_score"].notna()
    out["low"] = out["review_score"] <= config.LOW_REVIEW_MAX_SCORE
    out["written_before_arrival"] = out["delivered"] & (ans < dl)
    out["overdue_eod"] = (out["delivered"] & (dl >= out["eod"])) | (out["never"] & (out["eod"] <= last_ts))
    out["promise_reach"] = out["overdue_eod"] & ~(ans < out["eod"])
    out["delivery_reach"] = out["delivered"] & ~(ans < dl)
    out["reach_class"] = np.select(
        [out["promise_reach"] & out["delivery_reach"], out["promise_reach"], out["delivery_reach"]],
        [REACH_EITHER, REACH_PROMISE, REACH_DELIVERY], REACH_NEITHER)
    out["outcome"] = np.select([out["never"], out["late_calendar"], out["late"]],
                               [OUT_NEVER, OUT_LATE, OUT_LATE_DAY], OUT_ON_TIME)
    delivered = out[out["delivered"]]
    require(bool((delivered["late"] == (delivered["is_late"] == 1)).all()),
            "late must equal data_load's is_late on every delivered order")
    return out


def top_k_per_group(frame: pd.DataFrame, group: str, k: int | None,
                    score: str = "p_low_review") -> tuple[pd.Series, pd.Series]:
    """Stable rank (1 = riskiest) within each group, and whether it is in the top k.
    The stable sort on the frame's own row order is the tie-break review_timing.py uses."""
    q = frame.sort_values([group, score], ascending=[True, False], kind="stable")
    rank = (q.groupby(group).cumcount() + 1).reindex(frame.index)
    return rank, (rank <= k) if k is not None else pd.Series(True, index=frame.index)


def build_lists(r: pd.DataFrame, b1: pd.DataFrame, thr_slides: float, thr_deployed: float) -> dict:
    """Every contact list, as a boolean mask over its frame. `r` is the reviewed test
    orders in the row order of predictions_p2_test.csv; `b1` the delivered test orders in
    the row order of predictions_p1b_test.csv."""
    L = {}
    _, top = top_k_per_group(r, "purchase_day", 10)
    L["slides_ten_a_day"] = dict(frame="r", mask=top, moment="delivery", grouping="purchase day", k=10,
                                 pool=pd.Series(True, index=r.index),
                                 pool_name="reviewed test orders, grouped by the day they were bought "
                                           "and scored with their realised delivery (the slides' list)")
    L["slides_threshold_011"] = dict(frame="r", mask=r["p_low_review"] >= thr_slides, moment="delivery",
                                     grouping="none", k=None, pool=pd.Series(True, index=r.index),
                                     pool_name="reviewed test orders; threshold 0.11 maximises the net value "
                                               "on the test window itself (cost_p2.csv)")
    L["deployed_threshold_019"] = dict(frame="r", mask=r["p_low_review"] >= thr_deployed, moment="delivery",
                                       grouping="none", k=None, pool=pd.Series(True, index=r.index),
                                       pool_name="reviewed test orders; threshold 0.19 fixed on validation "
                                                 "(model_metadata.json)")
    pool = r["delivery_reach"]
    for k in DELIVERY_DAY_K:
        rank, _ = top_k_per_group(r[pool], "delivered_day", k)
        L[f"delivery_day_k{k}"] = dict(frame="r", mask=(rank <= k).reindex(r.index, fill_value=False),
                                       moment="delivery", grouping="delivery day", k=k, pool=pool,
                                       pool_name="delivered reviewed test orders whose customer had not yet "
                                                 "rated the order when the parcel arrived, grouped by delivery day")
    L["promised_date_message"] = dict(frame="r", mask=r["promise_reach"], moment="end of the promised day",
                                      grouping="none", k=None, pool=r["promise_reach"],
                                      pool_name="reviewed test orders with no parcel when the promised day ended "
                                                "and no review yet then (every one is contacted; no model)")
    q90 = float(b1["p_late_derived"].quantile(0.9))
    L["late_top_decile"] = dict(frame="b1", mask=b1["p_late_derived"] >= q90, moment="checkout",
                                grouping="none", k=None, pool=pd.Series(True, index=b1.index),
                                pool_name=f"delivered test orders; the riskiest tenth by late risk at checkout "
                                          f"(p_late_derived >= {q90:.3f})", q90=q90)
    return L


def list_table(L: dict, r: pd.DataFrame, b1: pd.DataFrame, n_days: int) -> pd.DataFrame:
    rows = []
    n_rev = len(r)
    for name, s in L.items():
        frame = r if s["frame"] == "r" else b1
        pos = frame["low"] if s["frame"] == "r" else frame["actual_late"] == 1
        sel, pool = s["mask"], s["pool"]
        hits = sel & pos
        row = {"list": name, "moment": s["moment"], "grouping": s["grouping"], "k_per_day": s["k"],
               "pool": s["pool_name"], "pool_n": int(pool.sum()),
               "pool_base_rate": float(pos[pool].mean()), "pool_positives": int((pool & pos).sum()),
               "contacts": int(sel.sum()), "hits": int(hits.sum())}
        if s["grouping"] == "delivery day":
            day = frame.loc[sel, "delivered_day"]
        elif s["grouping"] == "purchase day":
            day = frame.loc[sel, "purchase_day"]
        else:
            day = None
        if day is not None:
            per = day.value_counts()
            full = per.index[per == s["k"]]
            fm = sel & frame[("delivered_day" if s["grouping"] == "delivery day" else "purchase_day")].isin(full)
            row.update({"days": int(day.nunique()), "full_days": int(len(full)),
                        "precision_full_days": float(pos[fm].mean())})
        else:
            row.update({"days": n_days, "full_days": np.nan, "precision_full_days": np.nan})
        row["contacts_per_day"] = row["contacts"] / row["days"]
        row["precision"] = row["hits"] / row["contacts"]
        row["lift"] = row["precision"] / row["pool_base_rate"]
        row["recall"] = row["hits"] / row["pool_positives"]
        for cls in REACH_CLASSES:
            row[f"hits_{REACH_TAG[cls]}"] = (int((hits & (frame["reach_class"] == cls)).sum())
                                             if s["frame"] == "r" else np.nan)
        row["contacts_per_1k_reviewed"] = 1000 * row["contacts"] / n_rev if s["frame"] == "r" else np.nan
        rows.append(row)
    cols = ["list", "moment", "grouping", "k_per_day", "pool", "pool_n", "pool_base_rate", "pool_positives",
            "contacts", "days", "contacts_per_day", "full_days", "precision", "precision_full_days", "lift",
            "recall", "hits", "hits_promise_only", "hits_delivery_only", "hits_either", "hits_neither",
            "contacts_per_1k_reviewed"]
    return pd.DataFrame(rows)[cols]


SCOPES = ("all hits", "not written before arrival", "reachable at its moment")


def scope_hits(name: str, s: dict, r: pd.DataFrame) -> dict:
    """Which of a list's 1-2 star customers count: all of them; those who had not written
    before the parcel arrived (the slides' 'reachable'); or those the list's own contact
    could still reach - at delivery for the review-risk lists, when the promised day
    ended for the promised-date message."""
    sel = s["mask"]
    at_moment = r["promise_reach"] if name == "promised_date_message" else r["delivery_reach"]
    return {"all hits": sel & r["low"],
            "not written before arrival": sel & r["low"] & ~r["written_before_arrival"],
            "reachable at its moment": sel & r["low"] & at_moment}


def net_per_1k(hits: float, contacts: float, n: int, c_action: float, c_miss: float, efficacy: float) -> float:
    return 1000 * (efficacy * hits * c_miss - contacts * c_action) / n


def value_table(L: dict, r: pd.DataFrame, c_action: float, c_miss: float) -> pd.DataFrame:
    rows = []
    n = len(r)
    for name, s in L.items():
        if s["frame"] != "r":
            continue
        contacts = int(s["mask"].sum())
        for scope, m in scope_hits(name, s, r).items():
            hits = int(m.sum())
            prec = hits / contacts
            row = {"list": name, "scope": scope, "contacts": contacts, "hits": hits, "precision": prec}
            for e in EFFICACIES:
                row[f"net_e{int(100 * e)}"] = net_per_1k(hits, contacts, n, c_action, c_miss, e)
            row["breakeven_share"] = c_action / (c_miss * prec) if prec > 0 else np.inf
            for e in EFFICACIES:
                row[f"breakeven_review_cost_e{int(100 * e)}"] = c_action / (e * prec) if prec > 0 else np.inf
            rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_table(L: dict, r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(r)
    for name, s in L.items():
        if s["frame"] != "r":
            continue
        contacts = int(s["mask"].sum())
        for scope, m in scope_hits(name, s, r).items():
            hits = int(m.sum())
            prec = hits / contacts
            for c_call in SENS_CALL_COST:
                for c_rev in SENS_REVIEW_COST:
                    for e in SENS_EFFICACY:
                        rows.append({"list": name, "scope": scope, "precision": prec, "contacts": contacts,
                                     "call_cost": c_call, "review_cost": c_rev, "efficacy": e,
                                     "net_per_1k_reviewed": net_per_1k(hits, contacts, n, c_call, c_rev, e),
                                     "break_even_review_cost": c_call / (e * prec) if prec > 0 else np.inf})
    return pd.DataFrame(rows)


def decomposition_table(L: dict, r: pd.DataFrame, c_action: float, c_miss: float) -> pd.DataFrame:
    """The slides' values split by whom the contact would have reached: a delivered
    customer who had not yet written, a customer whose order never came, or a customer
    who had written while waiting for a parcel that then arrived late."""
    rows = []
    n = len(r)
    for name in ("slides_threshold_011", "deployed_threshold_019", "slides_ten_a_day"):
        sel = L[name]["mask"]
        h = sel & r["low"]
        groups = {"delivered_not_yet_written": h & r["delivered"] & ~r["written_before_arrival"],
                  "never_delivered": h & r["never"],
                  "late_written_before_arrival": h & r["delivered"] & r["written_before_arrival"]}
        contacts = int(sel.sum())
        row = {"list": name, "contacts": contacts}
        row.update({f"hits_{k}": int(v.sum()) for k, v in groups.items()})
        gross = {k: 1000 * int(v.sum()) * c_miss / n for k, v in groups.items()}
        row.update({f"gross_per_1k_{k}": v for k, v in gross.items()})
        cost = 1000 * contacts * c_action / n
        row["contact_cost_per_1k"] = cost
        row["net_all_per_1k"] = sum(gross.values()) - cost
        row["net_not_written_per_1k"] = gross["delivered_not_yet_written"] + gross["never_delivered"] - cost
        row["net_delivery_day_only_per_1k"] = gross["delivered_not_yet_written"] - cost
        row["late_never_gross_share_pct"] = (100 * (gross["never_delivered"] + gross["late_written_before_arrival"])
                                             / sum(gross.values()))
        row["written_before_arrival_on_time_hits"] = int((groups["late_written_before_arrival"] & ~r["late"]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def pilot_arms(lists: pd.DataFrame, c_action: float, c_miss: float) -> pd.DataFrame:
    """One arm per contact, sized on the customers that contact can still reach; plus the
    design on slide 11 for comparison. Two-sided two-proportion z-test, alpha 0.05
    (pilot_design.n_per_arm); weeks = ceil(arms * n / listed per day / 7)."""
    lt = lists.set_index("list")
    be_abs = c_action / c_miss
    msg, call, ten = lt.loc["promised_date_message"], lt.loc["delivery_day_k10"], lt.loc["slides_ten_a_day"]
    p_b, p_a, p_s = float(msg["precision"]), float(call["precision"]), float(ten["precision"])
    specs = [
        ("arm_b_two", "promised-date message: every customer whose promised day ends without a parcel and who "
                      "has not yet rated the order; message vs current practice", p_b,
         float(msg["contacts_per_day"]), 2, PILOT_TARGET),
        ("arm_b_three", "the same customers; message with the survey once the order is complete, message with "
                        "the survey unchanged, current practice", p_b, float(msg["contacts_per_day"]), 3,
         PILOT_TARGET),
        ("arm_a", "delivery-day call: each day's ten riskiest delivered parcels whose customer has not yet "
                  "rated the order, five contacted; sized for the break-even effect", p_a,
         float(PILOT_LISTED_PER_DAY), 2, be_abs / p_a),
        ("arm_a_25", "delivery-day call as above, sized for a one-in-four effect (reference)", p_a,
         float(PILOT_LISTED_PER_DAY), 2, PILOT_TARGET),
        ("slide_design", "slide 11: each day's top ten of the slides' list (purchase-day groups, scored with "
                         "the realised delivery), five called and five held back", p_s,
         float(PILOT_LISTED_PER_DAY), 2, PILOT_TARGET),
    ]
    rows = []
    for arm, population, p0, per_day, arms, target in specs:
        p1 = p0 * (1 - target)
        row = {"arm": arm, "population": population, "control_rate": p0, "per_day": per_day, "arms": arms,
               "breakeven_relative": be_abs / p0, "target_relative": target, "treated_rate": p1}
        for power in PD.POWERS:
            tag = f"{int(100 * power)}"
            n = PD.n_per_arm(p0, p1, power=power)
            row[f"n_per_arm_{tag}"] = n
            row[f"weeks_{tag}"] = float(np.ceil(arms * n / per_day / 7))
        rows.append(row)
    return pd.DataFrame(rows)


def outcomes_test(t: pd.DataFrame, r: pd.DataFrame, b1: pd.DataFrame, top10: pd.Series,
                  thr_prewarn: float) -> pd.DataFrame:
    """What happened to the customers of the test window. Every row names its population:
    the 12,927 orders placed, or the 12,630 delivered (the late warning at checkout)."""
    n = len(t)
    low = t["low"]
    in_top = t["order_id"].isin(set(r.loc[top10, "order_id"]))
    hit = in_top & low
    cls = t["reach_class"]
    spec = [
        ("placed", "orders placed (purchases 30 June to 30 August 2018)", pd.Series(True, index=t.index)),
        ("on_time", "parcel arrived by the promised date", t["delivered"] & ~t["late"]),
        ("late", "parcel arrived after the promised date began (the report's definition)", t["late"]),
        ("late_on_promised_day", "  ... during the promised day itself", t["late"] & ~t["late_calendar"]),
        ("late_after_promised_day", "  ... after the promised day ended", t["late_calendar"]),
        ("never_delivered", "parcel never delivered within the extract", t["never"]),
        ("never_delivered_cancelled", "  ... order cancelled or unavailable",
         t["never"] & t["order_status"].isin(["canceled", "unavailable"])),
        ("overdue_eod", "no parcel when the promised day ended", t["overdue_eod"]),
        ("overdue_eod_never", "  ... of whom never received it", t["overdue_eod"] & t["never"]),
        ("low", "rated the order 1-2 stars", low),
        ("low_on_time", "  ... parcel on time", low & t["delivered"] & ~t["late"]),
        ("low_late", "  ... parcel late", low & t["late"]),
        ("low_never", "  ... parcel never delivered", low & t["never"]),
        ("low_late_before_arrival", "late parcel, 1-2 stars written before it arrived",
         low & t["late"] & t["written_before_arrival"]),
        ("eod_low_not_yet_written", "no parcel when the promised day ended, not yet rated then, "
                                    "went on to rate 1-2 stars", low & t["promise_reach"]),
        ("reach_promise_only", "rated 1-2 stars; reachable only by the promised-date message",
         low & (cls == REACH_PROMISE)),
        ("reach_delivery_only", "rated 1-2 stars; reachable only by a call at delivery",
         low & (cls == REACH_DELIVERY)),
        ("reach_either", "rated 1-2 stars; reachable by either contact", low & (cls == REACH_EITHER)),
        ("reach_neither", "rated 1-2 stars before either moment", low & (cls == REACH_NEITHER)),
        ("top10", "on the slides' daily top-ten list (grouped by purchase day)", in_top),
        ("top10_hit", "  ... and rated the order 1-2 stars", hit),
        ("top10_hit_promise_reachable", "  ... of whom not yet rated when the promised day ended "
                                        "(no parcel then)", hit & t["promise_reach"]),
        ("top10_hit_delivery_only", "  ... of whom reachable only by a call at delivery",
         hit & (cls == REACH_DELIVERY)),
    ]
    rows = []
    for key, label, mask in spec:
        k = int(mask.sum())
        lo, hi = wilson(k, n)
        rows.append({"key": key, "outcome": label, "orders": k, "denominator": "placed", "denominator_n": n,
                     "per_1000": 1000 * k / n, "ci95_lo": 1000 * lo, "ci95_hi": 1000 * hi})
    nd = len(b1)
    flag = b1["p_late_derived"] >= thr_prewarn
    late = b1["actual_late"] == 1
    for key, label, mask in [
            ("prewarn_flagged", "warned at checkout (late risk at or above the 11% line)", flag),
            ("prewarn_late_flagged", "  ... parcel then late", flag & late),
            ("prewarn_late_missed", "late parcel, not warned", ~flag & late),
            ("prewarn_ontime_flagged", "warned, parcel then on time", flag & ~late)]:
        k = int(mask.sum())
        lo, hi = wilson(k, nd)
        rows.append({"key": key, "outcome": label, "orders": k, "denominator": "delivered", "denominator_n": nd,
                     "per_1000": 1000 * k / nd, "ci95_lo": 1000 * lo, "ci95_hi": 1000 * hi})
    return pd.DataFrame(rows)


# ===========================================================================
# E. The presentation's order, and one row per reviewed test order
# ===========================================================================
def journeys_table(r: pd.DataFrame, b_r1: float) -> pd.DataFrame:
    j = pd.DataFrame({
        "order_id": r["order_id"],
        "purchase": r["order_purchase_timestamp"],
        "promised": r["promised"],
        "shipping_limit": r["shipping_limit"],
        "carrier_handoff": r["order_delivered_carrier_date"],
        "delivered": r["order_delivered_customer_date"],
        "survey_sent": r["review_creation_date"],
        "answered": r["review_answer_timestamp"],
        "review_score": r["review_score"].astype(int),
        "review_title": r["review_comment_title"],
        "review_message": r["review_comment_message"],
        "order_status": r["order_status"],
        "customer_city": r["customer_city"], "customer_state": r["customer_state"],
        "customer_region": r["customer_region"],
        "seller_city": r["seller_city"], "seller_state": r["lead_seller_state"],
        "lead_category": r["lead_category"], "n_items": r["n_items"], "n_sellers": r["n_sellers"],
        "price": r["total_price"], "freight": r["total_freight"], "distance_km": r["distance_km"],
        "promised_days": r["estimated_days"],
        "actual_days": r["delivery_days"], "days_late": r["delay_vs_estimate_days"],
        "pred_days": r["pred_days"], "p_late_derived": r["p_late_derived"],
        "late_rank_on_day": r["late_rank_on_day"], "delivered_on_day": r["delivered_on_day"],
        "p_low_review": r["p_low_review"],
        "low_rank_purchase_day": r["low_rank_purchase_day"], "reviewed_on_purchase_day": r["reviewed_on_purchase_day"],
        "low_rank_delivery_day": r["low_rank_delivery_day"], "unrated_delivered_on_day": r["unrated_delivered_on_day"],
        "outcome": r["outcome"], "low": r["low"],
        "written_before_arrival": r["written_before_arrival"], "overdue_eod": r["overdue_eod"],
        "promise_reach": r["promise_reach"], "delivery_reach": r["delivery_reach"],
        "reach_class": r["reach_class"],
        "requoted_date": requote(r["order_purchase_timestamp"], r["pred_days"], b_r1),
        "review_predates_order": r["review_creation_date"] < r["order_purchase_timestamp"].dt.normalize(),
    })
    for c in ("n_items", "n_sellers", "late_rank_on_day", "delivered_on_day", "low_rank_purchase_day",
              "reviewed_on_purchase_day", "low_rank_delivery_day", "unrated_delivered_on_day"):
        j[c] = j[c].astype("Int64")
    return j


def story_tables(j: pd.DataFrame, r: pd.DataFrame, top10: pd.Series, b_r1: float,
                 story_ref: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    oid = story_ref["order_id"]
    s = j.set_index("order_id").loc[oid]
    x = r.set_index("order_id").loc[oid]
    day_mask = r["purchase_day"] == x["purchase_day"]
    day_top = r[day_mask & top10]
    hit = day_top[day_top["low"]]
    eod = s["promised"] + pd.Timedelta(days=1)
    story = pd.DataFrame([{
        "order_id": oid,
        "purchase": s["purchase"], "promised": s["promised"], "promised_days": s["promised_days"],
        "price": s["price"], "freight": s["freight"], "category": s["lead_category"],
        "customer_city": s["customer_city"], "customer_state": s["customer_state"],
        "seller_city": s["seller_city"], "seller_state": s["seller_state"], "distance_km": s["distance_km"],
        "pred_days_at_checkout": s["pred_days"], "p_late_at_checkout": s["p_late_derived"],
        "late_rank_on_day": int(s["late_rank_on_day"]), "delivered_orders_on_day": int(s["delivered_on_day"]),
        "shipping_limit": s["shipping_limit"], "carrier_handoff": s["carrier_handoff"],
        "handover_days_after_limit": (s["carrier_handoff"] - s["shipping_limit"]).total_seconds() / 86400,
        "survey_sent": s["survey_sent"], "review_answered": s["answered"], "review_score": int(s["review_score"]),
        "review_title": s["review_title"], "review_message": s["review_message"],
        "delivered": s["delivered"], "actual_days": s["actual_days"], "days_late": s["days_late"],
        "review_days_before_arrival": (s["delivered"] - s["answered"]).total_seconds() / 86400,
        "p_low_at_delivery": s["p_low_review"], "low_rank_on_day": int(s["low_rank_purchase_day"]),
        "reviewed_orders_on_day": int(s["reviewed_on_purchase_day"]),
        "requoted_date": s["requoted_date"],
        "requoted_missed_by_days": (s["delivered"].normalize() - s["requoted_date"]).days,
        "message_at": eod,
        "message_days_before_review": (s["answered"] - eod).total_seconds() / 86400,
        "call_days_after_review": (s["delivered"] - s["answered"]).total_seconds() / 86400,
        "reach_class": s["reach_class"],
        "top10_hits": len(hit),
        "top10_hits_late_written_before": int((hit["late"] & hit["written_before_arrival"]).sum()),
        "top10_hits_never_delivered": int(hit["never"].sum()),
        "top10_hits_reachable_at_delivery": int(hit["delivery_reach"].sum()),
        "top10_hits_reachable_on_promise": int(hit["promise_reach"].sum()),
    }])
    st = story.iloc[0]
    # every field story_order.csv already carries must agree with it
    for col in ("purchase", "delivered", "review_answered"):
        require(st[col] == pd.Timestamp(story_ref[col]), f"story {col} differs from story_order.csv")
    for col in ("seller_city", "seller_state", "customer_city", "customer_state", "category"):
        require(st[col] == story_ref[col], f"story {col} differs from story_order.csv")
    for col, ref in (("p_late_at_checkout", "p_late_at_checkout"), ("p_low_at_delivery", "p_low_review_at_delivery"),
                     ("pred_days_at_checkout", "pred_days_at_checkout"), ("distance_km", "distance_km"),
                     ("days_late", "days_late"), ("review_days_before_arrival", "review_days_before_arrival")):
        require(abs(float(st[col]) - float(story_ref[ref])) < 1e-9, f"story {col} differs from story_order.csv")
    require(st["low_rank_on_day"] == int(story_ref["rank_on_day"])
            and st["reviewed_orders_on_day"] == int(story_ref["orders_on_day"])
            and st["top10_hits"] == int(story_ref["top10_low_reviews"])
            and int(len(hit) - hit["delivery_reach"].sum()) == int(story_ref["top10_written_before_arrival"])
            + int(hit["never"].sum()),
            "story ranks and top-ten counts differ from story_order.csv")
    when_pct = lambda p: f"{100 * p:.1f}%"
    ev = [
        (st["purchase"], "customer",
         f"Order placed: {st['category']}, R${st['price']:.2f} + R${st['freight']:.2f} freight, from "
         f"{st['seller_city']} ({st['seller_state']}) to {st['customer_city']} ({st['customer_state']}), "
         f"{st['distance_km']:,.0f} km; delivery promised by {st['promised']:%d %b %Y} "
         f"({st['promised_days']:.1f} days)", "order_purchase_timestamp; order_estimated_delivery_date"),
        (st["purchase"], "platform and models",
         f"Checkout forecast {st['pred_days_at_checkout']:.1f} days; risk of missing the promised date "
         f"{when_pct(st['p_late_at_checkout'])}, #{st['late_rank_on_day']} of {st['delivered_orders_on_day']} "
         f"delivered orders bought that day",
         "predictions_p1a_test.csv pred_days; predictions_p1b_test.csv p_late_derived"),
        (st["shipping_limit"], "seller and carrier", "Seller's deadline to hand the parcel to the carrier",
         "order_items.shipping_limit_date"),
        (st["carrier_handoff"], "seller and carrier",
         f"Parcel handed to the carrier, {st['handover_days_after_limit']:.1f} days after the seller's deadline",
         "orders.order_delivered_carrier_date"),
        (st["promised"], "customer", "Promised date: no parcel", "orders.order_estimated_delivery_date"),
        (st["message_at"], "platform and models",
         "The promised day ends without a parcel; the tables record no message to the customer "
         "(the moment of the recommended promised-date message)", "order_estimated_delivery_date + 1 day"),
        (st["survey_sent"], "platform and models", "Satisfaction survey sent", "order_reviews.review_creation_date"),
        (st["review_answered"], "customer",
         f"Customer rates the order {st['review_score']} star{'s' if st['review_score'] != 1 else ''}, "
         f"{st['review_days_before_arrival']:.1f} days before the parcel arrives",
         "order_reviews.review_answer_timestamp; review_comment_title; review_comment_message"),
        (st["requoted_date"], "platform and models",
         f"Date the forecast plus the {b_r1:g}-day margin would have quoted at checkout",
         "customer_promise_rules.csv (R1 margin) + predictions_p1a_test.csv pred_days"),
        (st["delivered"], "seller and carrier",
         f"Parcel delivered, {st['days_late']:.1f} days after the promised date "
         f"({st['actual_days']:.1f} days after checkout)", "orders.order_delivered_customer_date"),
        (st["delivered"], "platform and models",
         f"Review-risk score at delivery {when_pct(st['p_low_at_delivery'])}, #{st['low_rank_on_day']} of "
         f"{st['reviewed_orders_on_day']} reviewed orders bought that day; the review was already "
         f"{st['review_days_before_arrival']:.1f} days old", "predictions_p2_test.csv p_low_review"),
    ]
    events = pd.DataFrame(ev, columns=["when", "lane", "what", "source_column"])
    events["when"] = pd.to_datetime(events["when"])
    events = events.sort_values("when", kind="stable").reset_index(drop=True)
    events.insert(0, "order_id", oid)
    events.insert(1, "step", np.arange(1, len(events) + 1))
    events.insert(3, "day_offset", (events["when"] - st["purchase"]).dt.total_seconds() / 86400)
    return story, events[["order_id", "step", "when", "day_offset", "lane", "what", "source_column"]]


# ===========================================================================
def main(out_dir: Path = RESULTS, labels_file: Path = LABELS_FILE) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not labels_file.exists():
        raise SystemExit(f"{labels_file} is missing: it is the committed hand reading of the topic patterns.")
    tables = load_all()
    orders = build_orders(tables)
    window = orders[orders["order_purchase_timestamp"].between(config.ANALYSIS_START, config.ANALYSIS_END)]
    df = build_features(window.reset_index(drop=True), with_history=True)
    last_ts = orders[["order_purchase_timestamp", "order_delivered_customer_date"]].max().max()
    c = config.COSTS
    c_action, c_miss = c["outreach_cost"], c["low_review_cost"]
    res: dict[str, pd.DataFrame] = {}

    # -- A. what unhappy customers wrote -------------------------------------
    rv = review_text(df, tables["order_reviews"])
    res["topics"] = complaint_topics(rv)
    res["precision"] = complaint_precision(rv, pd.read_csv(labels_file))
    res["dictionary"] = complaint_dictionary()
    res["basket"] = basket_table(rv)
    res["timing"] = survey_timing(df)
    res["lateness"] = lateness_days(df)

    # -- B. do let-down customers come back ----------------------------------
    res["repeat"] = repeat_table(first_orders(orders))

    # -- C. a date the customer can rely on ----------------------------------
    pred = score_lead_time(df)
    tvt = need("train_val_test.csv")
    val_mae_ref = float(tvt.loc[(tvt["problem"] == "P1A lead time") & (tvt["period"] == "validation"), "mae"].iloc[0])
    val = pred[pred["period"] == "validation"]
    val_mae = float(np.abs(val["pred_days"] - val["delivery_days"]).mean())
    require(abs(val_mae - val_mae_ref) < 1e-9, f"validation MAE {val_mae} differs from train_val_test.csv {val_mae_ref}")
    rules, region, frontier = promise_tables(pred)
    rules["n_late"] = (rules["late_pct"] * rules["n"] / 100).round().astype(int)
    frontier = frontier_validation(pred, frontier)
    b_r1 = float(rules.loc[rules["rule"].str.startswith("R1"), "buffer_days"].iloc[0])
    olist_val = float(rules.loc[(rules["rule"] == "Olist's own promise") & (rules["period"] == "validation"),
                                "late_pct"].iloc[0])
    require(choose_buffer(frontier) == b_r1 and choose_buffer(frontier, olist_val) == b_r1,
            "choose_buffer() on the frontier's validation columns must give the R1 margin")
    res.update(rules=rules, region=region, frontier=frontier)

    # -- D. the customers of the test window ---------------------------------
    t = customer_flags(df[df["order_purchase_timestamp"] > config.VAL_END].copy(), last_ts)
    items = tables["order_items"]
    t["shipping_limit"] = t["order_id"].map(items.groupby("order_id")["shipping_limit_date"].max())
    t["seller_city"] = t["lead_seller_id"].map(tables["sellers"].set_index("seller_id")["seller_city"])
    text = rv.set_index("order_id")[["review_comment_title", "review_comment_message"]]
    t = t.join(text, on="order_id")

    p2 = need("predictions_p2_test.csv", parse_dates=["order_purchase_timestamp"])
    p1a = need("predictions_p1a_test.csv")
    b1 = need("predictions_p1b_test.csv", parse_dates=["order_purchase_timestamp"]).dropna(subset=["actual_late"])
    require(set(p2["order_id"]) == set(t.loc[t["reviewed"], "order_id"]),
            "predictions_p2_test.csv must hold exactly the reviewed test orders")
    require(set(b1["order_id"]) == set(t.loc[t["delivered"], "order_id"]) == set(p1a["order_id"]),
            "predictions_p1a/p1b_test.csv must hold exactly the delivered test orders")
    # Reviewed test orders in the row order of predictions_p2_test.csv: the stable sorts
    # below then break ties exactly as review_timing.py does.
    r = p2[["order_id", "actual_low_review", "p_low_review"]].merge(t, on="order_id", how="left",
                                                                    validate="one_to_one")
    require(bool((r["order_id"].to_numpy() == p2["order_id"].to_numpy()).all())
            and bool((r["order_purchase_timestamp"].to_numpy() == p2["order_purchase_timestamp"].to_numpy()).all()),
            "the reviewed frame must keep predictions_p2_test.csv's rows and order")
    require(bool((r["low"] == (r["actual_low_review"] == 1)).all()), "1-2 star flags disagree with predictions_p2")
    r["purchase_day"] = r["order_purchase_timestamp"].dt.date
    r["delivered_day"] = r["order_delivered_customer_date"].dt.date
    r["pred_days"] = r["order_id"].map(p1a.set_index("order_id")["pred_days"])
    b1["purchase_day"] = b1["order_purchase_timestamp"].dt.date
    b1["late_rank_on_day"], _ = top_k_per_group(b1, "purchase_day", None, score="p_late_derived")
    b1["delivered_on_day"] = b1.groupby("purchase_day")["order_id"].transform("size")
    b1i = b1.set_index("order_id")
    r["p_late_derived"] = r["order_id"].map(b1i["p_late_derived"])
    r["late_rank_on_day"] = r["order_id"].map(b1i["late_rank_on_day"])
    r["delivered_on_day"] = r["purchase_day"].map(b1.groupby("purchase_day")["order_id"].size())
    r["low_rank_purchase_day"], _ = top_k_per_group(r, "purchase_day", None)
    r["reviewed_on_purchase_day"] = r.groupby("purchase_day")["order_id"].transform("size")
    pool = r["delivery_reach"]
    rank_d, _ = top_k_per_group(r[pool], "delivered_day", None)
    r["low_rank_delivery_day"] = rank_d.reindex(r.index)
    r["unrated_delivered_on_day"] = r.loc[pool].groupby("delivered_day")["order_id"].transform("size").reindex(r.index)

    meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text("utf-8"))
    cost = need("cost_p2.csv")
    thr_slides = float(cost.loc[cost["net_benefit"].idxmax(), "threshold"])
    require(round(thr_slides, 2) == 0.11, f"the slides' cost-optimal threshold moved to {thr_slides}")
    thr_deployed = float(meta["p2"]["threshold"])
    thr_prewarn = float(meta["p1b"]["threshold"])
    L = build_lists(r, b1, thr_slides, thr_deployed)
    n_days = int(r["purchase_day"].nunique())

    # the reproductions the slides and the existing tables rest on
    top10 = L["slides_ten_a_day"]["mask"]
    require(int(top10.sum()) == 614 and int((top10 & r["low"]).sum()) == 351,
            "the daily top-ten list no longer reproduces 614 contacts / 351 hits (review_timing.py)")
    comp = need("review_timing_composition.csv").set_index("list").loc["all reviewed orders"]
    require(int(comp["contacts"]) == int(top10.sum()) and int(comp["hits"]) == int((top10 & r["low"]).sum()),
            "the daily top-ten list differs from review_timing_composition.csv")
    m011 = L["slides_threshold_011"]["mask"]
    row011 = cost.loc[cost["net_benefit"].idxmax()]
    require(int(m011.sum()) == int(row011["flagged"]) == 1927 and int((m011 & r["low"]).sum()) == int(row011["tp"]) == 712,
            "threshold 0.11 no longer reproduces cost_p2.csv's 1,927 contacts / 712 hits")

    lists = list_table(L, r, b1, n_days)
    res["lists"] = lists
    res["value"] = value_table(L, r, c_action, c_miss)
    res["sensitivity"] = sensitivity_table(L, r)
    dec = decomposition_table(L, r, c_action, c_miss)
    res["decomposition"] = dec

    tv = need("review_timing_value.csv")
    for name, thr in (("slides_threshold_011", 0.11), ("deployed_threshold_019", 0.19)):
        ref = tv.loc[(tv["threshold"] - thr).abs() < 1e-9].iloc[0]
        d = dec.set_index("list").loc[name]
        require(abs(d["net_all_per_1k"] - ref["net_all_per_1k"]) < 1e-6
                and abs(d["net_not_written_per_1k"] - ref["net_reachable_per_1k"]) < 1e-6,
                f"{name}: the decomposition does not reproduce review_timing_value.csv")
    d11, d19 = dec.set_index("list").loc["slides_threshold_011"], dec.set_index("list").loc["deployed_threshold_019"]
    require(round(d11["net_all_per_1k"]) == 1466 and round(d11["net_not_written_per_1k"]) == 775,
            "threshold 0.11 must reproduce R$1,466 / R$775")
    require(round(d19["net_all_per_1k"]) == 1373 and round(d19["net_not_written_per_1k"]) == 692,
            "threshold 0.19 must reproduce R$1,373 / R$692")
    cap = need("review_timing_capacity.csv")
    ten = cap[(cap["kind"] == "daily") & (cap["k"] == "10 a day")].iloc[0]
    per_1k = 1000 * ten["contacts"] / len(need("review_timing_test_orders.csv"))
    ten_all = per_1k * (ten["precision"] * c_miss - c_action)
    ten_reach = per_1k * (ten["reachable_precision"] * c_miss - c_action)
    dten = dec.set_index("list").loc["slides_ten_a_day"]
    require(abs(dten["net_all_per_1k"] - ten_all) < 1e-6 and abs(dten["net_not_written_per_1k"] - ten_reach) < 1e-6
            and round(dten["net_all_per_1k"]) == 853 and round(dten["net_not_written_per_1k"]) == 433,
            "the ten-a-day list must reproduce phase3_stats' R$853 / R$433")

    res["outcomes"] = outcomes_test(t, r, b1, top10, thr_prewarn)
    o = res["outcomes"].set_index("key")["orders"]
    require(int(o["low"]) == int(r["low"].sum()) == 1418, "1-2 star test customers must be 1,418")
    require(sum(int(o[k]) for k in ("reach_promise_only", "reach_delivery_only", "reach_either", "reach_neither"))
            == int(o["low"]), "the reach classes must sum to the 1-2 star customers")
    require(int(o["on_time"]) + int(o["late"]) + int(o["never_delivered"]) == int(o["placed"]),
            "on time + late + never delivered must be every order placed")
    require(int(o["prewarn_late_flagged"]) + int(o["prewarn_late_missed"]) == int(o["late"]),
            "the warning's late parcels must be the late parcels")
    res["pilot"] = pilot_arms(lists, c_action, c_miss)
    pdz = need("pilot_design.csv")
    ref = pdz[(pdz["k_per_day"] == 10) & ((pdz["relative_reduction"] - PILOT_TARGET).abs() < 1e-9)].iloc[0]
    sd = res["pilot"].set_index("arm").loc["slide_design"]
    require(abs(sd["control_rate"] - ref["control_rate"]) < 1e-12 and sd["n_per_arm_80"] == ref["n_per_arm_80"] == 191
            and sd["weeks_80"] == ref["weeks_80"] and sd["n_per_arm_90"] == ref["n_per_arm_90"]
            and sd["weeks_90"] == ref["weeks_90"], "the slide-11 design must equal pilot_design.csv")

    # -- E. the presentation's order and every reviewed test order -------------
    res["journeys"] = journeys_table(r, b_r1)
    story_ref = need("story_order.csv").iloc[0]
    story, events = story_tables(res["journeys"], r, top10, b_r1, story_ref)
    require(story.loc[0, "order_id"] == story_ref["order_id"], "story order differs from story_order.csv")
    res["story"], res["story_events"] = story, events

    files = {"topics": "customer_complaint_topics.csv", "precision": "customer_complaint_precision.csv",
             "dictionary": "customer_complaint_dictionary.csv", "basket": "customer_basket.csv",
             "timing": "customer_survey_timing.csv", "lateness": "customer_lateness_days.csv",
             "repeat": "customer_repeat.csv", "rules": "customer_promise_rules.csv",
             "region": "customer_promise_region.csv", "frontier": "customer_promise_frontier.csv",
             "outcomes": "customer_outcomes_test.csv", "lists": "customer_contact_lists.csv",
             "value": "customer_value.csv", "decomposition": "customer_value_decomposition.csv",
             "sensitivity": "customer_value_sensitivity.csv", "pilot": "customer_pilot_arms.csv",
             "story": "customer_story.csv", "story_events": "customer_story_events.csv",
             "journeys": "customer_journeys_test.csv"}
    for key, name in files.items():
        res[key].to_csv(out_dir / name, index=False)
    res["_last_ts"] = pd.DataFrame({"last_ts": [last_ts]})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--labels", type=Path, default=LABELS_FILE)
    a = ap.parse_args()
    res = main(a.out, a.labels)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    for k in ("outcomes", "lists", "value", "decomposition", "pilot", "rules", "lateness", "precision"):
        print(f"\n=== {k}")
        print(res[k].round(4).to_string(index=False))
    print("\n=== story")
    print(res["story"].T.to_string())
    print(f"\nlast timestamp in the extract: {res['_last_ts'].iloc[0, 0]}")
    print(f"wrote customer_*.csv into {a.out}")
