# app.py
# Streamlit UI para brincar com heurísticas do rectpack
# Corrige bugs do código original (escopo, tipagem, e mapeamento de unidades)

import io
from typing import Dict, Tuple, List, Any

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rectpack import newPacker, MaxRectsBssf, GuillotineBlsfSas, SkylineMwfl


# -------------------------
# Core helpers (corrigidos)
# -------------------------

def scale_bin_dims(width_mm: float, height_mm: float, margin_mm: float, scale: float) -> Tuple[int, int]:
    """Converte dimensões da folha (mm) para inteiros (unidades do packer), descontando margens."""
    bw = int((width_mm - 2 * margin_mm) * scale)
    bh = int((height_mm - 2 * margin_mm) * scale)
    if bw <= 0 or bh <= 0:
        raise ValueError("Dimensões úteis do bin são não positivas. Verifique largura/altura e margens.")
    return bw, bh


def scale_labels_mm_to_int(
    labels_mm: List[Tuple[float, float, int]],  # [(altura_mm, largura_mm, qtd)]
    spacing_mm: float,
    scale: float
) -> Dict[Tuple[int, int], int]:
    """
   Transforma rótulos em inteiros (unidades do packer).
    Importante: rectpack usa (w, h). Aqui tratamos largura como w, altura como h.
    Adiciona o espaçamento em ambos os eixos antes de escalar, para garantir folgas.
    """
    out: Dict[Tuple[int, int], int] = {}
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


def pack_once(
    algorithm_cls: Any,
    labels_scaled: Dict[Tuple[int, int], int],
    bin_w: int,
    bin_h: int,
    rotation: bool
):
    """
    Executa o packing para uma heurística.
    Retorna lista de colocações [(x_mm, y_mm, w_mm, h_mm)], mais métricas.
    """
    packer = newPacker(pack_algo=algorithm_cls, rotation=rotation)

    # Adiciona retângulos
    total_rects = 0
    area_rects_scaled = 0
    for (w, h), q in labels_scaled.items():
        for _ in range(q):
            packer.add_rect(w, h)
        total_rects += q
        area_rects_scaled += w * h * q

    # Um único bin
    packer.add_bin(bin_w, bin_h, count=1)
    packer.pack()

    placements = [p for p in packer.rect_list() if p[0] == 0]  # bin index 0
    placed = len(placements)

    return placements, placed, total_rects, area_rects_scaled


def placements_to_mm(
    placements,
    scale: float,
    margin_mm: float,
    spacing_mm: float
) -> List[Tuple[float, float, float, float]]:
    """Converte as posições do rectpack (int) para mm, descontando spacing nos lados."""
    out = []
    for (_b, x, y, w, h, _rid) in placements:
        x_mm = x / scale + margin_mm
        y_mm = y / scale + margin_mm
        w_mm = w / scale - spacing_mm
        h_mm = h / scale - spacing_mm
        out.append((x_mm, y_mm, w_mm, h_mm))
    return out


def fig_from_layout(
    placements_mm: List[Tuple[float, float, float, float]],
    page_w_mm: float,
    page_h_mm: float,
    margin_mm: float,
    title: str
):
    """Desenha folha, área útil e rótulos em mm."""
    fig, ax = plt.subplots(figsize=(8, 8 * (page_h_mm / page_w_mm)))

    # Rótulos
    for (x, y, w, h) in placements_mm:
        rect = Rectangle((x, y), w, h, edgecolor="blue", facecolor="skyblue", alpha=0.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{w:.0f}×{h:.0f}", ha="center", va="center", fontsize=8)

    # Área útil (margens)
    ax.add_patch(Rectangle((margin_mm, margin_mm),
                           page_w_mm - 2 * margin_mm,
                           page_h_mm - 2 * margin_mm,
                           fill=False, ls="--", ec="red", lw=1, label="Área útil"))

    # Folha
    ax.add_patch(Rectangle((0, 0), page_w_mm, page_h_mm, fill=False, ec="black", lw=1, label="Folha"))

    ax.set_xlim(0, page_w_mm)
    ax.set_ylim(0, page_h_mm)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()  # Conveniente para visualizar (0,0) no canto superior
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def utilization_stats(
    placements_mm: List[Tuple[float, float, float, float]],
    page_w_mm: float,
    page_h_mm: float,
    margin_mm: float
) -> Dict[str, float]:
    """Calcula métricas de utilização com base na área útil em mm²."""
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

st.title("RectPack Playground (Streamlit)")
st.caption("Interface simples para experimentar heurísticas (MaxRects, Guillotine, Skyline) com parâmetros de folha, margem, espaçamento e escala.")

with st.sidebar:
    st.header("Parâmetros da Folha (mm)")
    page_w = st.number_input("Largura da folha (mm)", value=485.0, min_value=1.0, step=1.0)
    page_h = st.number_input("Altura da folha (mm)", value=500.0, min_value=1.0, step=1.0)

    st.header("Layout")
    margin = st.number_input("Margem (mm)", value=7.5, min_value=0.0, step=0.5)
    spacing = st.number_input("Espaçamento entre rótulos (mm)", value=4.3, min_value=0.0, step=0.1)

    st.header("Escala")
    scale = st.number_input("Fator de escala (px/mm)", value=10.0, min_value=0.1, step=0.1, help="rectpack exige inteiros; multiplicamos mm por 'scale' e arredondamos para inteiro.")

    st.header("Heurísticas")
    alg_map = {
        "MaxRectsBssf": MaxRectsBssf,
        "GuillotineBlsfSas": GuillotineBlsfSas,
        "SkylineMwfl": SkylineMwfl,
    }
    alg_choices = st.multiselect("Selecione heurísticas", list(alg_map.keys()), default=["MaxRectsBssf", "GuillotineBlsfSas", "SkylineMwfl"])
    rotation = st.toggle("Permitir rotação", value=True)

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
        # Extrai e valida rótulos
        labels_input: List[Tuple[float, float, int]] = []
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

        # Gera resultados por heurística
        for alg_name in alg_choices:
            algorithm_cls = alg_map[alg_name]

            placements, placed, total_rects, area_rects_scaled = pack_once(
                algorithm_cls, labels_scaled, bin_w, bin_h, rotation
            )
            placements_mm = placements_to_mm(placements, scale, margin, spacing)
            stats = utilization_stats(placements_mm, page_w, page_h, margin)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = fig_from_layout(placements_mm, page_w, page_h, margin, title=f"Heurística: {alg_name}")
                st.pyplot(fig, clear_figure=True)

            with col2:
                st.markdown(f"### Métricas — {alg_name}")
                st.write(f"Retângulos colocados: **{placed}** de **{total_rects}**")
                st.write(f"Área útil (mm²): **{stats['area_util_mm2']:.0f}**")
                st.write(f"Área colocada (mm²): **{stats['area_colocada_mm2']:.0f}**")
                st.write(f"Utilização da área útil: **{stats['utilizacao_percent']:.2f}%**")

                # Download do PNG
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    label=f"Baixar PNG ({alg_name})",
                    data=buf,
                    file_name=f"grid_{alg_name}.png",
                    mime="image/png"
                )

    except Exception as e:
        st.error(f"Falha ao gerar layout: {e}")
