import io
import math
from typing import Any

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from rectpack import newPacker, MaxRectsBssf, GuillotineBlsfSas, SkylineMwfl

def scale_bin_dims(width_mm: float, height_mm: float, margin_mm: float, scale: float) -> tuple[int, int]:
    bw = int((width_mm - 2 * margin_mm) * scale)
    bh = int((height_mm - 2 * margin_mm) * scale)
    if bw <= 0 or bh <= 0:
        raise ValueError("Dimensões úteis do bin são não positivas. Verifique largura/altura e margens.")
    return bw, bh


def scale_labels_mm_to_int(labels_mm: list[tuple[float, float, int]], spacing_mm: float, scale: float) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for altura, largura, qtd in labels_mm:
        if qtd <= 0:
            continue
        w_scaled = int((largura + spacing_mm) * scale)
        h_scaled = int((altura + spacing_mm) * scale)
        if w_scaled <= 0 or h_scaled <= 0:
            raise ValueError("Algum rótulo virou não positivo após a escala. Revise medidas/escala/espaçamento.")
        out[(w_scaled, h_scaled)] = out.get((w_scaled, h_scaled), 0) + int(qtd)
    if not out:
        raise ValueError("Nenhum rótulo válido informado.")
    return out


def labels_scaled_area(labels_scaled: dict[tuple[int, int], int]) -> int:
    return sum(w * h * q for (w, h), q in labels_scaled.items())

def pack_min_bins_auto(algorithm_cls: Any,labels_scaled: dict[tuple[int, int], int],bin_w: int,bin_h: int,rotation: bool,max_bins: int = 256) -> tuple[list, int, int]:
    """
    Tenta alocar todos os retângulos. Se não couberem, aumenta nº de bins e repete.
    Retorna:
      - placements: lista [(bin_id, x, y, w, h, rid)]
      - used_bins: int (nº de bins adicionados na solução)
      - total_rects: int
    """
    total_rects = sum(labels_scaled.values())
    if total_rects == 0:
        raise ValueError("Nenhum retângulo para alocar.")

    rects_area = labels_scaled_area(labels_scaled)
    bin_area = bin_w * bin_h

    count = max(1, math.ceil(rects_area / bin_area))
    while count <= max_bins:
        packer = newPacker(pack_algo=algorithm_cls, rotation=rotation)
        for (w, h), q in labels_scaled.items():
            for _ in range(q):
                packer.add_rect(w, h)
        for _ in range(count):
            packer.add_bin(bin_w, bin_h)

        packer.pack()

        placements = packer.rect_list()
        placed = len(placements)

        if placed >= total_rects:
            return placements, count, total_rects

        count = min(max_bins, count * 2)

    raise RuntimeError(
        f"Não foi possível alocar todos os retângulos até o limite de {max_bins} folhas. "
        "Reduza espaçamentos/aumente a folha ou aumente o limite."
    )


def placements_grouped_mm(placements,scale: float,margin_mm: float,spacing_mm: float) -> dict[int, list[tuple[float, float, float, float]]]:
    """
    Converte para mm e agrupa por bin_id.
    Retorna {bin_id: [(x_mm, y_mm, w_mm, h_mm), ...]}
    """
    grouped: dict[int, list[tuple[float, float, float, float]]] = {}
    for (b, x, y, w, h, _rid) in placements:
        x_mm = x / scale + margin_mm
        y_mm = y / scale + margin_mm
        w_mm = w / scale - spacing_mm
        h_mm = h / scale - spacing_mm
        grouped.setdefault(b, []).append((x_mm, y_mm, w_mm, h_mm))
    return grouped


# -------------------------
# Visualização e métricas
# -------------------------

def fig_from_layout(placements_mm: list[tuple[float, float, float, float]],page_w_mm: float,page_h_mm: float,margin_mm: float,title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8 * (page_h_mm / page_w_mm)))

    for (x, y, w, h) in placements_mm:
        rect = Rectangle((x, y), w, h, edgecolor="blue", facecolor="skyblue", alpha=0.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{w:.0f}×{h:.0f}", ha="center", va="center", fontsize=8)

    # Área útil
    ax.add_patch(Rectangle((margin_mm, margin_mm),
                           page_w_mm - 2 * margin_mm,
                           page_h_mm - 2 * margin_mm,
                           fill=False, ls="--", ec="red", lw=1, label="Área útil"))

    # Folha
    ax.add_patch(Rectangle((0, 0), page_w_mm, page_h_mm, fill=False, ec="black", lw=1, label="Folha"))

    ax.set_xlim(0, page_w_mm)
    ax.set_ylim(0, page_h_mm)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def utilization_stats(
    placements_mm: list[tuple[float, float, float, float]],
    page_w_mm: float,
    page_h_mm: float,
    margin_mm: float
) -> dict[str, float]:
    usable_w = page_w_mm - 2 * margin_mm
    usable_h = page_h_mm - 2 * margin_mm
    usable_area = max(usable_w, 0) * max(usable_h, 0)
    placed_area = sum(max(w, 0) * max(h, 0) for (_x, _y, w, h) in placements_mm)
    util = (placed_area / usable_area * 100) if usable_area > 0 else 0.0
    return {
        "area_util_mm2": usable_area,
        "area_colocada_mm2": placed_area,
        "utilizacao_percent": util
    }


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="RectPack Playground", layout="wide")

st.title("Simulador de Layouts")

with st.sidebar:
    st.header("Folha (mm)")
    page_w = st.number_input("Largura da folha (mm)", min_value=1.0, step=1.0)
    page_h = st.number_input("Altura da folha (mm)", min_value=1.0, step=1.0)

    st.header("Layout")
    margin = st.number_input("Margem (mm)", value=7.5, min_value=0.0, step=0.5)
    spacing = st.number_input("Espaçamento entre rótulos (mm)", value=4.3, min_value=0.0, step=0.1)

    st.header("Escala")
    scale = st.number_input("Fator de escala (px/mm)", value=10.0, min_value=0.1, step=0.1)

    st.header("Heurísticas")
    alg_map = {
        "MaxRectsBssf": MaxRectsBssf,
        "GuillotineBlsfSas": GuillotineBlsfSas,
        "SkylineMwfl": SkylineMwfl,
    }
    alg_choices = st.multiselect("Selecione heurísticas", list(alg_map.keys()), default=["MaxRectsBssf", "GuillotineBlsfSas", "SkylineMwfl"])
    rotation = st.toggle("Permitir rotação", value=True)

    st.header("Limites")
    max_bins = st.number_input("Máximo de folhas (proteção)", value=64, min_value=1, step=1,
                               help="Evita travar se as medidas forem inviáveis. A alocação para quando atingir esse número.")

st.subheader("Rótulos (mm)")
st.write("Edite a tabela: altura, largura e quantidade.")

default_rows = [
    {"altura_mm": 29.0,  "largura_mm": 53.4, "qtd": 4},
    {"altura_mm": 142.6, "largura_mm": 75.7, "qtd": 4},
    {"altura_mm": 90.3,  "largura_mm": 24.0, "qtd": 4},
    {"altura_mm": 170.6, "largura_mm": 146.4, "qtd": 4},
]
df = st.data_editor(
    default_rows,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "altura_mm": st.column_config.NumberColumn("Altura (mm)", step=0.1, min_value=0.0),
        "largura_mm": st.column_config.NumberColumn("Largura (mm)", step=0.1, min_value=0.0),
        "qtd": st.column_config.NumberColumn("Qtd", step=1, min_value=0),
    }
)

btn = st.button("Gerar layouts", type="primary")

if btn:
    try:
        labels_input: list[tuple[float, float, int]] = []
        for row in df:
            try:
                a = float(row["altura_mm"])
                l = float(row["largura_mm"])
                q = int(row["qtd"])
            except Exception:
                continue
            if a > 0 and l > 0 and q > 0:
                labels_input.append((a, l, q))
        if not labels_input:
            st.error("Informe ao menos um rótulo válido (altura > 0, largura > 0, qtd > 0).")
            st.stop()

        # Escala bin e rótulos
        bin_w, bin_h = scale_bin_dims(page_w, page_h, margin, scale)
        labels_scaled = scale_labels_mm_to_int(labels_input, spacing, scale)
        total_rects_expected = sum(q for (_a, _l, q) in labels_input)

        # Gera resultados por heurística
        for alg_name in alg_choices:
            algorithm_cls = alg_map[alg_name]

            placements, used_bins, total_rects = pack_min_bins_auto(algorithm_cls, labels_scaled, bin_w, bin_h, rotation, max_bins=int(max_bins))
            if total_rects != total_rects_expected:
                st.info(f"Aviso: retângulos agregados por (w,h) iguais após escala. {total_rects} retângulos no pack vs {total_rects_expected} na tabela.")

            grouped_mm = placements_grouped_mm(placements, scale, margin, spacing)
            bin_ids_sorted = sorted(grouped_mm.keys())

            st.markdown(f"## Heurística: {alg_name} — {len(bin_ids_sorted)} folha(s) usada(s)")

            # Abas por folha
            tabs = st.tabs([f"Folha {i+1}" for i in range(len(bin_ids_sorted))])
            for idx, b in enumerate(bin_ids_sorted):
                with tabs[idx]:
                    placements_mm = grouped_mm[b]
                    fig = fig_from_layout(placements_mm, page_w, page_h, margin, title=f"{alg_name} — Folha {idx+1}/{len(bin_ids_sorted)}"
                    )
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.pyplot(fig, clear_figure=True)
                    with col2:
                        stats = utilization_stats(placements_mm, page_w, page_h, margin)
                        st.markdown(f"**Métricas (Folha {idx+1})**")
                        st.write(f"Área útil (mm²): **{stats['area_util_mm2']:.0f}**")
                        st.write(f"Área colocada (mm²): **{stats['area_colocada_mm2']:.0f}**")
                        st.write(f"Utilização da área útil: **{stats['utilizacao_percent']:.2f}%**")

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                    buf.seek(0)
                    st.download_button(
                        label=f"Baixar PNG (Folha {idx+1})",
                        data=buf,
                        file_name=f"grid_{alg_name}_folha_{idx+1}.png",
                        mime="image/png",
                        key=f"png_{alg_name}_{idx}"
                    )

    except Exception as e:
        st.error(f"Falha ao gerar layout: {e}")
