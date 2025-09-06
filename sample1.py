import io
import math
from typing import Any

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from rectpack import newPacker, MaxRectsBssf, GuillotineBlsfSas, SkylineMwfl


#-------------------------#
# Definições e Auxiliares #
#-------------------------#

def scale_bin_dims(width_mm: float, height_mm: float, margin_mm: float, scale: float) -> tuple[int, int]:

    """O pacote rectpack não aceita valores float, sendo necessário uma scala para adequar valores de bins quebrados"""

    bw = int((width_mm - 2 * margin_mm) * scale)
    bh = int((height_mm - 2 * margin_mm) * scale)
    if bw <= 0 or bh <= 0:
        raise ValueError("Dimensões úteis do bin são não positivas. Verifique largura/altura e margens.")
    return bw, bh


def scale_labels(labels_mm: list[tuple[float, float, int]], spacing_mm: float, scale: float) -> dict[tuple[int, int], int]:

    """O pacote rectpack não aceita valores float, sendo necessário uma scala para adequar valores dos rótulos quebrados"""

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


def scaler(labels_scaled: dict[tuple[int, int], int]) -> int:
    return sum(w * h * q for (w, h), q in labels_scaled.items())


#-------------------#
# Multi-Bin Packing #
#-------------------#

def recursive_packer(algorithm_cls: Any,labels_scaled: dict[tuple[int, int], int],bin_w: int,bin_h: int,rotation: bool,max_bins: int = 256):
    """
    Tenta alocar todos os retângulos recursivamente
    Retorna:
      - placements: lista [(bin_id, x, y, w, h, rid)]
      - used_bins: int (nº de bins adicionados na solução)
      - total_rects: int
    """
    total_rects = sum(labels_scaled.values())
    if total_rects == 0:
        raise ValueError("Nenhum retângulo para alocar.")

    rects_area = scaler(labels_scaled)
    bin_area = bin_w * bin_h
    lower_bound = max(1, math.ceil(rects_area / bin_area))

    count = lower_bound
    while count <= max_bins:
        packer = newPacker(pack_algo=algorithm_cls, rotation=rotation)

        for (w, h), q in labels_scaled.items():
            for _ in range(q):
                packer.add_rect(w, h)

        for _ in range(count):
            packer.add_bin(bin_w, bin_h)

        packer.pack() #type: ignore

        placements = packer.rect_list()  # [(bin_id, x, y, w, h, rid)]
        placed = len(placements)

        if placed >= total_rects:
            return placements, count, total_rects

        count = min(max_bins, count * 2)

    raise RuntimeError(
        f"Não foi possível alocar todos os {total_rects} retângulos até o limite de {max_bins} folhas. "
        "Aumente a folha, reduza espaçamentos, permita rotação, ou suba o limite."
    )

def placements_grouped_mm(placements: list, scale:float, margin_mm: float, spacing_mm: float) -> dict[int, list[tuple[float, float, float, float]]]:

    """Retorno: [(bin_id, x, y, w, h, rid)]"""

    grouped: dict[int, list[tuple[float, float, float, float]]] = {}
    for (b, x, y, w, h, _rid) in placements:
        x_mm = x / scale + margin_mm
        y_mm = y / scale + margin_mm
        w_mm = w / scale - spacing_mm
        h_mm = h / scale - spacing_mm
        grouped.setdefault(b, []).append((x_mm, y_mm, w_mm, h_mm))
    return grouped


#--------------------------#
# Visualização e Benckmark #
#--------------------------#

def fig_from_layout(placements_mm: list[tuple[float, float, float, float]],page_w_mm: float,page_h_mm: float,margin_mm: float,title: str):

    """Cria um plot que simula o layout de impressão"""

    fig, ax = plt.subplots(figsize=(8, 8 * (page_h_mm / page_w_mm)))

    for (x, y, w, h) in placements_mm:
        rect = Rectangle((x, y), w, h, edgecolor="blue", facecolor="skyblue", alpha=0.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{w:.1f}×{h:.1f}", ha="center", va="center", fontsize=8)

    ax.add_patch(Rectangle((margin_mm, margin_mm),
                           page_w_mm - 2 * margin_mm,
                           page_h_mm - 2 * margin_mm,
                           fill=False, ls="--", ec="red", lw=1, label="Área útil"))

    ax.add_patch(Rectangle((0, 0), page_w_mm, page_h_mm, fill=False, ec="black", lw=1, label="Folha"))

    ax.set_xlim(0, page_w_mm)
    ax.set_ylim(0, page_h_mm)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def benchmark(placements_mm: list[tuple[float, float, float, float]],page_w_mm: float,page_h_mm: float,margin_mm: float) -> dict[str, float]:

    """Métricas para se determinar o quão efetivo"""

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


#------------------#
# Streamlit Layout #
#------------------#

st.set_page_config(page_title="RectPack Playground", layout="wide")
st.title("Simulador de Empacotamento 2D - Estudo de caso em uma Gráfica de rótulos adesivos")
with st.sidebar:
    st.header("Folha (mm)")
    page_w = st.number_input("Largura da folha (mm)", value=485.0, min_value=1.0, step=1.0)
    page_h = st.number_input("Altura da folha (mm)", value=500.0, min_value=1.0, step=1.0)

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
    alg_choices = st.multiselect("Selecione uma ou mais herísticas heurísticas", list(alg_map.keys()),
                                 default=["MaxRectsBssf", "GuillotineBlsfSas", "SkylineMwfl"])
    
    rotation = st.toggle("Permitir rotação", value=True)

    st.header("Limites")
    max_bins = st.number_input("Máximo de folhas (proteção)", value=64, min_value=1, step=1)

    st.header("Modo")
    mode = st.select_slider(
        "Modo de empacotamento",
        options=["Agrupado", "Único"],
        value="Agrupado",
        help='Modo "Agrupado" traz todos os rótulos junto no mesmo grid. Modo "Único" aplica todas as heurísticas escolhidas em cada objeto'
    )

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

        bin_w, bin_h = scale_bin_dims(page_w, page_h, margin, scale)
        labels_scaled = scale_labels(labels_input, spacing, scale)

        for alg_name in alg_choices:
            algorithm_cls = alg_map[alg_name]
            st.markdown(f"## Heurística: {alg_name}")

            if mode == "Agrupado":
                #MODO AGRUPADO 
                placements, used_bins, total_rects = recursive_packer(algorithm_cls, labels_scaled, bin_w, bin_h, rotation, max_bins=int(max_bins))

                grouped_mm = placements_grouped_mm(placements, scale, margin, spacing)
                bin_ids_sorted = sorted(grouped_mm.keys())

                rows = []
                for b in bin_ids_sorted:
                    for (x, y, w, h) in grouped_mm[b]:
                        rows.append({"bin_id": b, "x_mm": x, "y_mm": y, "w_mm": w, "h_mm": h})
                csv_df = pd.DataFrame(rows, columns=["bin_id", "x_mm", "y_mm", "w_mm", "h_mm"])
                st.download_button(
                    label=f"Baixar CSV ({alg_name}) — modo agrupado",
                    data=csv_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"grid_{alg_name}_agrupado.csv",
                    mime="text/csv"
                )

                tabs = st.tabs([f"Folha {i+1}" for i in range(len(bin_ids_sorted))])
                for idx, b in enumerate(bin_ids_sorted):
                    with tabs[idx]:
                        placements_mm = grouped_mm[b]
                        fig = fig_from_layout(
                            placements_mm, page_w, page_h, margin,
                            title=f"{alg_name} — Folha {idx+1}/{len(bin_ids_sorted)} (Agrupado)"
                        )
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.pyplot(fig, clear_figure=True,width=500)
                        with col2:
                            stats = benchmark(placements_mm, page_w, page_h, margin)
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
                            file_name=f"grid_{alg_name}_agrupado_folha_{idx+1}.png",
                            mime="image/png",
                            key=f"png_{alg_name}_agr_{idx}"
                        )

            else:
                #MODO ÚNICO
                def type_label_from_scaled(w_int: int, h_int: int) -> str:
                    w_mm_real = w_int / scale - spacing
                    h_mm_real = h_int / scale - spacing
                    return f"{w_mm_real:.1f}×{h_mm_real:.1f} mm"

                type_keys_sorted = sorted(labels_scaled.keys(), key=lambda wh: (wh[0]*wh[1], wh[0], wh[1]))
                outer_tabs = st.tabs([f"{type_label_from_scaled(*wh)}" for wh in type_keys_sorted])


                all_rows = []
                for idx_type, wh in enumerate(type_keys_sorted):
                    w_int, h_int = wh
                    q = labels_scaled[wh]
                    label_txt = type_label_from_scaled(w_int, h_int)

                    placements, used_bins, total_rects = recursive_packer(algorithm_cls, {wh: q}, bin_w, bin_h, rotation, max_bins=int(max_bins))

                    grouped_mm = placements_grouped_mm(placements, scale, margin, spacing)
                    bin_ids_sorted = sorted(grouped_mm.keys())

                    with outer_tabs[idx_type]:
                        st.markdown(f"**Tipo:** {label_txt} — **Qtd:** {q} — **Folhas usadas:** {len(bin_ids_sorted)}")

                        inner_tabs = st.tabs([f"Folha {i+1}" for i in range(len(bin_ids_sorted))])
                        for idx_bin, b in enumerate(bin_ids_sorted):
                            with inner_tabs[idx_bin]:
                                placements_mm = grouped_mm[b]
                                fig = fig_from_layout(
                                    placements_mm, page_w, page_h, margin,
                                    title=f"{alg_name} — {label_txt} — Folha {idx_bin+1}/{len(bin_ids_sorted)} (Único por tipo)"
                                )

                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.pyplot(fig, clear_figure=True, width=500)
                                with col2:
                                    stats = benchmark(placements_mm, page_w, page_h, margin)
                                    st.markdown(f"**Métricas (Folha {idx_bin+1})**")
                                    st.write(f"Área útil (mm²): **{stats['area_util_mm2']:.0f}**")
                                    st.write(f"Área colocada (mm²): **{stats['area_colocada_mm2']:.0f}**")
                                    st.write(f"Utilização da área útil: **{stats['utilizacao_percent']:.2f}%**")

                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                                buf.seek(0)
                                st.download_button(
                                    label=f"Baixar PNG ({label_txt}) — Folha {idx_bin+1}",
                                    data=buf,
                                    file_name=f"grid_{alg_name}_unico_{label_txt.replace('×','x').replace(' ','')}_folha_{idx_bin+1}.png",
                                    mime="image/png",
                                    key=f"png_{alg_name}_unico_{idx_type}_{idx_bin}"
                                )

                                for (x, y, w, h) in placements_mm:
                                    all_rows.append({
                                        "tipo_w_mm": float(f"{w_int/scale - spacing:.3f}"),
                                        "tipo_h_mm": float(f"{h_int/scale - spacing:.3f}"),
                                        "bin_id": int(b),
                                        "x_mm": x, "y_mm": y, "w_mm": w, "h_mm": h
                                    })

                if all_rows:
                    csv_df = pd.DataFrame(all_rows, columns=["tipo_w_mm", "tipo_h_mm", "bin_id", "x_mm", "y_mm", "w_mm", "h_mm"])
                    st.download_button(
                        label=f"Baixar CSV ({alg_name}) — modo único por tipo",
                        data=csv_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"grid_{alg_name}_unico_por_tipo.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Falha ao gerar layout: {e}")


