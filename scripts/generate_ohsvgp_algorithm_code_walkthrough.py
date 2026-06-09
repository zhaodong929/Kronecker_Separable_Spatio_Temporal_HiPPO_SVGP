from __future__ import annotations

import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "routeB_formula_code_walkthrough"
MD = OUTDIR / "ohsvgp_algorithm_code_walkthrough.md"
PDF = OUTDIR / "ohsvgp_algorithm_code_walkthrough.pdf"


def esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br/>")


def code_block(text: str, style: ParagraphStyle) -> Preformatted:
    return Preformatted(text.strip(), style)


def make_table(rows: list[list[str]], widths: list[float], style: ParagraphStyle) -> Table:
    table = Table(
        [[Paragraph(esc(cell), style) for cell in row] for row in rows],
        colWidths=[w * cm for w in widths],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfd8dc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


ALGO1_ROWS = [
    [
        "Require: X, y, Z, A(t), B(t), m_u, S_u",
        "时间/空间输入、目标、inducing 结构、HiPPO basis、变分参数。",
        "BatchKroneckerSTHiPPOSVGP.__init__ 初始化 temporal_builder、spatial_kernel、z_s、noise/covariate parameters；forward 接收 times, x_s, y。",
        "st_model_batch.py:83-128, 296-312；temporal_analytic.py:150-276",
        "部分一致。代码是时空 Kronecker 版本，不是截图中的 1D Z in R^{M x 1}。m_u/S_u 不是外部自由参数，而由 Gaussian 闭式后验求出。",
    ],
    [
        "1: K_fu = k(X,Z), K_uu = k(Z,Z), K_uu^{t1} from HiPPO ODEs",
        "构造训练点到 inducing 的 cross-covariance 和 inducing prior covariance；HiPPO basis 演化到最终时间。",
        "build_temporal_covariances 调 compute_kuu_t/compute_kfu_t；build_spatial_covariances 调 spatial kernel；temporal basis 由 spherical Bessel/RFF analytic builder 生成。",
        "st_model_batch.py:179-194；temporal_analytic.py:211-271",
        "实现了等价对象 Kfu/Kuu，但是 analytic HiPPO-RFF builder，不显式求解 A(t),B(t) ODE。",
    ],
    [
        "2: mu(x_i)=K_{f_i,u}^{t1}(K_uu^{t1})^{-1} m_u",
        "用 variational posterior mean 得到 latent mean。",
        "Projection A_t/A_s = Kfu Kuu^{-1}；train_mean = A_t M_u A_s^T + covariate_mean。",
        "st_model_batch.py:196-211, 330-334",
        "一致，扩展为 Kronecker spatio-temporal mean。",
    ],
    [
        "3: sigma^2(x_i)=K_ff - K_fu K_uu^{-1}[K_uu-S_u]K_uu^{-1}K_uf",
        "预测方差包含 prior diag、sparse projection residual、posterior covariance correction。",
        "predict 中 prior_diag - projected_prior_diag + posterior_correction；posterior_correction 用 posterior precision Cholesky。",
        "st_model_batch.py:405-440",
        "一致，代码以 precision form 计算 S_u 贡献。",
    ],
    [
        "4: ell_var_exp = sum E_q log p(y_i|f_i)",
        "ELBO expected likelihood 项。",
        "Gaussian likelihood 下被解析积分，gaussian_nll 用 logdet_kuu、logdet_precision、info_dot_mean 直接给 marginal/NLL objective。",
        "st_model_batch.py:336-347；kron_ops.py:143-157",
        "等价于 Gaussian closed-form objective；不是通用 quadrature/MC ELBO。",
    ],
    [
        "5: KL(q(u)||p(u))",
        "SVGP 的 variational KL 正则。",
        "没有单独显式 compute KL；Gaussian conjugate 情况下 KL/likelihood 项被合并进 closed-form precision posterior 与 gaussian_nll。",
        "st_model_batch.py:234-257, 336-347",
        "数学上被折叠进 closed-form Gaussian evidence/NLL；不适用于非 Gaussian likelihood 的通用 ELBO。",
    ],
    [
        "6: return ell_var_exp - KL",
        "最大化 ELBO。",
        "训练时返回 loss=gaussian_nll；优化器最小化该 loss。",
        "st_model_batch.py:341-379；tests/test_stage1.py:test_small_synthetic_training_reduces_loss",
        "符号相反但目标等价：min NLL 等价 max Gaussian ELBO/evidence。",
    ],
]


ALGO2_ROWS = [
    [
        "Require: X', y', Z_t1, Z_t2, A(t),B(t), m_{u_t1},S_{u_t1}, m_{u_t2},S_{u_t2}, K_{u_t1,u_t1}",
        "第二任务使用 task1 旧 inducing posterior 和 task2 新 inducing posterior；旧点冻结，新点更新。",
        "OnlinePosteriorState 存 reference_horizon, Kuu, inv_Kuu, lambda_precision, h, m；initialize 建立共享 reference inducing coordinate。",
        "st_model_online.py:24-42, 222-269",
        "不完全一致。代码没有同时维护截图中的 q(u_t1) 和 q(u_t2) 两套独立 variational parameters；而是维护一个 reference-coordinate posterior summary。",
    ],
    [
        "1: K_{f',u_t2}, K_{u_t2,u_t2}, K_{u_t2,u_t1}",
        "新任务 basis 与旧任务 basis 的 covariance/cross-covariance。",
        "update_block 计算 kfu_t_local, kuu_t_local；_temporal_transfer_matrix 计算 cross_kuu_t 和 residual_kuu_t。",
        "st_model_online.py:292-322, 504-508",
        "实现了 local/reference cross-covariance；命名是 local_horizon/reference_horizon，而非 t1/t2 variational pair。",
    ],
    [
        "2: mu_t2(x')=K_{f',u_t2}K_{u_t2,u_t2}^{-1}m_{u_t2}",
        "第二任务新 posterior mean。",
        "a_t_local = Kfu_local Kuu_local^{-1}；a_t = a_t_local @ temporal_transfer；posterior mean m 在 reference coordinate 中 recover。",
        "st_model_online.py:509-529, 532-534",
        "部分一致。mean 在 reference posterior summary 上计算；local basis 通过条件均值 map 回 reference basis。",
    ],
    [
        "3: sigma_t2^2(x') with S_{u_t2}",
        "第二任务 posterior variance。",
        "predict 中 prior_diag - projected_prior_diag + posterior_correction；posterior_correction 来自 Lambda^{-1}。",
        "st_model_online.py:581-613",
        "一致于 Gaussian precision posterior；但 S_{u_t2} 不是独立存储，而是 Lambda^{-1}。",
    ],
    [
        "4: ell_var_exp over new task",
        "新任务 expected log likelihood。",
        "update_block 使用 residual_grid 构造 delta_lambda/delta_h；block_nll 只作为诊断返回，不作为完整 ELBO correction objective。",
        "st_model_online.py:495-539",
        "部分实现。Gaussian update 等价于吸收新 block likelihood；没有显式返回 ell_var_exp - KL + CorrectionTerm。",
    ],
    [
        "5: m_t1t2 = K_{u_t1,u_t2} K_{u_t2,u_t2}^{-1} m_{u_t2}",
        "通过 p(u_t1|u_t2) 把新 posterior 投影回旧 inducing。",
        "_temporal_transfer_matrix 计算 E[u_local | u_ref] = cross_kuu_t @ inv_Kuu_ref；方向是 local -> reference summary 的条件均值近似。",
        "st_model_online.py:292-322",
        "方向和结构不是截图的逐字实现。代码采用 reference-coordinate map，不显式构造 m_t1t2。",
    ],
    [
        "6: S_t1t2 = ... covariance of q_t2(u_t1)",
        "把新 posterior 不确定性映射回旧 inducing，并加条件残差。",
        "代码计算 residual_kuu_t = K_local - K_cross K_ref^{-1} K_cross^T；该残差进入 _residual_corrected_summary_terms 的 observation covariance correction。",
        "st_model_online.py:315-378",
        "部分替代。代码没有形成截图中的完整 S_t1t2 对象，也没有用它计算两个 KL 差。",
    ],
    [
        "7: KL(q(u_t2)||p(u_t2))",
        "第二任务新 posterior 对新 prior 的 KL。",
        "没有显式 KL；Gaussian update 使用 precision accumulation。",
        "st_model_online.py:527-529, 380-384",
        "未按截图显式实现。",
    ],
    [
        "8: CorrectionTerm = KL(q_t1t2||p_t1) - KL(q_t1t2||q_t1)",
        "OHSVGP 防遗忘 correction term。",
        "代码中不存在 CorrectionTerm 的 KL 差实现；注释明确当前 online summary 不需要额外 residual correction term 的条件，且 residual correction 指的是条件残差协方差进 likelihood summary，不是 OHSVGP KL correction。",
        "st_model_online.py:386-399；grep correction term 只见此处注释",
        "未实现截图 Algorithm 2 的核心 correction-term ELBO。当前是 Gaussian posterior-summary 在线递推，不是原始 OHSVGP ELBO 逐项优化。",
    ],
    [
        "9: return ell_var_exp - KL + CorrectionTerm",
        "返回第二任务 OHSVGP ELBO。",
        "update_block 返回 Lambda, h, m, Kuu/Kfu/A_t, rmse, pred_nll 等诊断；没有返回 ELBO + CorrectionTerm。",
        "st_model_online.py:541-559",
        "未实现该返回形式。",
    ],
]


ROUTEB_ROWS = [
    [
        "截图 OHSVGP 的核心问题",
        "当 HiPPO basis 随任务/时间改变，旧 posterior q(u_t1) 如何转移并避免遗忘。",
        "Route B 代码不使用截图的 correction-term KL 差，而是转移旧 likelihood ratio q_{old}/p 到新坐标：B_old -> L_t^T B_old L_t，H_old -> H_old L_t，并在联合 beta-u precision 中累加。",
        "joint_ssgp_kron/model.py:304-384；ssgp_transfer.py:10-113",
        "这是项目后期实现的替代路线：更接近 streaming VFE/old-likelihood transfer，而不是 Algorithm 2 OHSVGP ELBO。",
    ],
    [
        "OHSVGP-style evaluation",
        "学习新 block 后评估旧 block 性能，考察 retained seen-history behavior。",
        "run_hipposvgp_era5_routeb.py 的 --ohsvgp-heldout-eval 保存 block-pair metrics M_{n,j}、forgetting curves。",
        "scripts/run_hipposvgp_era5_routeb.py:1391-1727, 2141-2255, 2503-2517",
        "这是评估协议仿照 OHSVGP 问题，不代表训练算法逐字实现 Algorithm 2。",
    ],
]


def markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    body = rows[1:]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        out.append("| " + " | ".join(cell.replace("\n", "<br>") for cell in row) + " |")
    return "\n".join(out)


def write_markdown() -> None:
    lines = [
        "# OHSVGP Algorithm 1/2 与项目代码实现对照",
        "",
        "本文档根据用户提供的截图中 Algorithm 1 和 Algorithm 2，逐行核对当前项目实现。",
        "",
        "## 总结结论",
        "",
        "- Algorithm 1 的单任务 HIPPO-SVGP/SVGP Gaussian 核心，在 `BatchKroneckerSTHiPPOSVGP` 中以 Kronecker spatio-temporal closed-form Gaussian posterior/NLL 的形式实现。",
        "- Algorithm 2 的完整 OHSVGP 第二任务 ELBO，尤其是 `CorrectionTerm(t1,t2)=KL(q_t1t2||p_t1)-KL(q_t1t2||q_t1)`，没有按截图逐项实现。",
        "- 当前 `OnlinePosteriorSummarySTGP` 是 Gaussian-conjugate online posterior-summary recursion：累加 precision `Lambda` 和 information vector `h`，并通过 conditional-mean transfer 把 local temporal basis 映射到 reference basis。",
        "- 项目中的 Route B 是另一条后续路线：转移旧 likelihood ratio / natural statistics，而不是实现截图中的 OHSVGP correction-term ELBO。",
        "",
        "## Algorithm 1 对照表",
        "",
        markdown_table([["截图步骤", "数学含义", "代码实现", "代码位置", "核对结论"]] + ALGO1_ROWS),
        "",
        "## Algorithm 2 对照表",
        "",
        markdown_table([["截图步骤", "数学含义", "代码实现", "代码位置", "核对结论"]] + ALGO2_ROWS),
        "",
        "## Route B 与 OHSVGP 的关系",
        "",
        markdown_table([["主题", "截图 OHSVGP", "项目实现", "代码位置", "核对结论"]] + ROUTEB_ROWS),
        "",
        "## 可复用伪代码：当前项目实际实现",
        "",
        "### A. 当前 Stage 1 batch Gaussian HiPPO-SVGP",
        "",
        "```text",
        "Input: times, spatial grid X_s, observations Y, temporal_config, spatial_kernel_config, spatial inducing Z_s",
        "1. Build temporal HiPPO-RFF basis for the horizon.",
        "2. Compute Kuu_t and Kfu_t from the temporal builder.",
        "3. Compute Kzz_s and Kxz_s from the spatial kernel.",
        "4. Form A_t = Kfu_t Kuu_t^{-1}, A_s = Kxz_s Kzz_s^{-1}.",
        "5. Form posterior precision Lambda = kron(Kuu_t^{-1}, Kzz_s^{-1}) + sigma^{-2} kron(A_t^T A_t, A_s^T A_s).",
        "6. Form information h = sigma^{-2} vec(A_t^T Y A_s).",
        "7. Solve m = Lambda^{-1} h.",
        "8. Optionally materialize S = Lambda^{-1}; compute train mean, NLL, RMSE.",
        "```",
        "",
        "### B. 当前 Stage 2 online posterior-summary update",
        "",
        "```text",
        "Input: reference horizon, fixed spatial inducing Z_s, online blocks B_1,...,B_N",
        "Initialize:",
        "  Lambda_0 = kron(Kuu_ref^{-1}, Kzz_s^{-1})",
        "  h_0 = 0, m_0 = 0",
        "For each block n:",
        "  1. Resolve local block horizon.",
        "  2. Compute Kfu_local, Kuu_local.",
        "  3. Compute temporal transfer T = K_local,ref K_ref,ref^{-1}.",
        "  4. Compute residual covariance R = K_local - K_local,ref K_ref^{-1} K_ref,local.",
        "  5. Compute local projection A_t_local = Kfu_local Kuu_local^{-1}.",
        "  6. Map to reference projection A_t = A_t_local T.",
        "  7. Compute A_s = Kxz_s Kzz_s^{-1}.",
        "  8. If R is zero: delta_Lambda = sigma^{-2} A^T A, delta_h = sigma^{-2} A^T y.",
        "     Else: compute residual-corrected delta_Lambda and delta_h using an observation covariance C.",
        "  9. Accumulate Lambda_n = Lambda_{n-1} + delta_Lambda, h_n = h_{n-1} + delta_h.",
        "  10. Recover m_n = Lambda_n^{-1} h_n.",
        "  11. Predict with prior_diag - projected_prior_diag + A_* Lambda_n^{-1} A_*^T.",
        "```",
        "",
        "### C. 截图 Algorithm 2 中当前没有逐项实现的部分",
        "",
        "```text",
        "Not implemented as written:",
        "  - Independent q(u_t2)=N(m_u_t2,S_u_t2) optimization objective.",
        "  - Explicit q_t1t2(u_t1)=int p(u_t1|u_t2)q_t2(u_t2)du_t2 object.",
        "  - Explicit KL(q(u_t2)||p(u_t2)).",
        "  - Explicit CorrectionTerm(t1,t2)=KL(q_t1t2||p(u_t1))-KL(q_t1t2||q(u_t1)).",
        "  - Return ell_var_exp - KL + CorrectionTerm.",
        "```",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def styles():
    registerFont(UnicodeCIDFont("STSong-Light"))
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=sample["Title"], fontName="STSong-Light", fontSize=18, leading=24, alignment=TA_CENTER),
        "h2": ParagraphStyle("h2", parent=sample["Heading2"], fontName="STSong-Light", fontSize=13, leading=17, spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("body", parent=sample["BodyText"], fontName="STSong-Light", fontSize=9.2, leading=13, alignment=TA_LEFT, spaceAfter=5),
        "small": ParagraphStyle("small", parent=sample["BodyText"], fontName="STSong-Light", fontSize=7.1, leading=9),
        "code": ParagraphStyle("code", parent=sample["Code"], fontName="Courier", fontSize=7.4, leading=9.2, backColor=colors.HexColor("#f6f8fa"), borderPadding=5),
    }


def build_pdf() -> None:
    st = styles()
    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=A4,
        leftMargin=1.25 * cm,
        rightMargin=1.25 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title="OHSVGP Algorithm Code Walkthrough",
    )
    story = [
        Paragraph("OHSVGP Algorithm 1/2 与项目代码实现对照", st["title"]),
        Spacer(1, 0.25 * cm),
        Paragraph("本文档根据截图中的 OHSVGP Algorithm 1/2，逐项核对当前项目代码。重点结论：Algorithm 1 的 Gaussian 单任务核心已实现；Algorithm 2 的完整 correction-term ELBO 未按截图逐字实现，当前代码采用 posterior-summary recursion 和 Route B old-likelihood transfer 作为替代路线。", st["body"]),
        Paragraph("1. 总结判断", st["h2"]),
        make_table(
            [
                ["项目部分", "是否匹配截图流程", "说明"],
                ["Stage 1 batch HIPPO-SVGP", "基本匹配 Algorithm 1", "以 Gaussian closed-form posterior/NLL 实现，扩展为 Kronecker 时空模型。"],
                ["Stage 2 OnlinePosteriorSummarySTGP", "部分匹配 Algorithm 2 的输入/预测思想", "有跨 horizon covariance 和 basis transfer，但没有显式 q_t1t2 与 correction-term KL。"],
                ["Route B structured_joint", "不是截图 Algorithm 2", "转移旧 likelihood natural statistics，是后续替代路线。"],
                ["OHSVGP-style ERA5 evaluation", "匹配评估问题，不匹配训练算法", "评估旧 block retained performance，但训练不是原始 OHSVGP ELBO。"],
            ],
            [4.0, 4.1, 9.0],
            st["small"],
        ),
        Paragraph("2. Algorithm 1 对照", st["h2"]),
        make_table([["截图步骤", "数学含义", "代码实现", "代码位置", "核对结论"]] + ALGO1_ROWS, [3.6, 3.5, 5.0, 3.1, 4.2], st["small"]),
        PageBreak(),
        Paragraph("3. Algorithm 2 对照", st["h2"]),
        make_table([["截图步骤", "数学含义", "代码实现", "代码位置", "核对结论"]] + ALGO2_ROWS, [3.7, 3.4, 5.1, 3.0, 4.3], st["small"]),
        PageBreak(),
        Paragraph("4. Route B 与 OHSVGP 的关系", st["h2"]),
        make_table([["主题", "截图 OHSVGP", "项目实现", "代码位置", "核对结论"]] + ROUTEB_ROWS, [3.3, 4.0, 5.7, 3.3, 3.7], st["small"]),
        Paragraph("5. 当前项目实际伪代码", st["h2"]),
        Paragraph("A. Stage 1 batch Gaussian HiPPO-SVGP", st["body"]),
        code_block(
            """
Input: times, spatial grid X_s, observations Y, temporal_config, spatial_kernel_config, spatial inducing Z_s
1. Build temporal HiPPO-RFF basis for the horizon.
2. Compute Kuu_t and Kfu_t from the temporal builder.
3. Compute Kzz_s and Kxz_s from the spatial kernel.
4. Form A_t = Kfu_t Kuu_t^{-1}, A_s = Kxz_s Kzz_s^{-1}.
5. Lambda = kron(Kuu_t^{-1}, Kzz_s^{-1}) + sigma^{-2} kron(A_t^T A_t, A_s^T A_s).
6. h = sigma^{-2} vec(A_t^T Y A_s).
7. Solve m = Lambda^{-1} h.
8. Compute train mean, Gaussian NLL, RMSE.
            """,
            st["code"],
        ),
        Paragraph("B. Stage 2 online posterior-summary update", st["body"]),
        code_block(
            """
Initialize:
  Lambda_0 = kron(Kuu_ref^{-1}, Kzz_s^{-1}), h_0 = 0, m_0 = 0
For each online block:
  1. Resolve local block horizon.
  2. Compute Kfu_local, Kuu_local.
  3. Compute transfer T = K_local,ref K_ref,ref^{-1}.
  4. Compute residual covariance R = K_local - K_local,ref K_ref^{-1} K_ref,local.
  5. A_t_local = Kfu_local Kuu_local^{-1}; A_t = A_t_local T.
  6. A_s = Kxz_s Kzz_s^{-1}.
  7. Build delta_Lambda and delta_h, with residual-corrected observation covariance if R != 0.
  8. Accumulate Lambda and h.
  9. Recover m = Lambda^{-1} h.
 10. Predict using prior_diag - projected_prior_diag + A_* Lambda^{-1} A_*^T.
            """,
            st["code"],
        ),
        Paragraph("C. 截图 Algorithm 2 未逐项实现的内容", st["body"]),
        code_block(
            """
Not implemented as written:
  - independent q(u_t2)=N(m_u_t2,S_u_t2) optimization objective;
  - explicit q_t1t2(u_t1)=int p(u_t1|u_t2)q_t2(u_t2)du_t2;
  - explicit KL(q(u_t2)||p(u_t2));
  - explicit CorrectionTerm(t1,t2)=KL(q_t1t2||p(u_t1))-KL(q_t1t2||q(u_t1));
  - return ell_var_exp - KL + CorrectionTerm.
            """,
            st["code"],
        ),
        Paragraph("6. 审查结论", st["h2"]),
        Paragraph("如果目标是复现截图中的原始 OHSVGP Algorithm 2，那么当前项目没有完整实现该 ELBO 流程，尤其缺失 correction-term 的两个 KL 项。当前代码更准确的名称应是 Kronecker HiPPO-SVGP Gaussian batch + online posterior-summary recursion，以及后续 Route B structured old-likelihood transfer。若论文/报告中使用 OHSVGP，应把它限定为 OHSVGP-style evaluation 或 related baseline inspiration，而不能声称代码逐字实现了截图 Algorithm 2。", st["body"]),
    ]

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("STSong-Light", 7.5)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(1.25 * cm, 0.8 * cm, "OHSVGP Algorithm 1/2 与项目代码实现对照")
        canvas.drawRightString(A4[0] - 1.25 * cm, 0.8 * cm, str(doc_obj.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_markdown()
    build_pdf()
    print(MD)
    print(PDF)


if __name__ == "__main__":
    main()
