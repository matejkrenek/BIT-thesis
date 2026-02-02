import os
import os.path as osp
from typing import Callable, List, Optional, Union
from tqdm import tqdm

import torch
import numpy as np
import open3d as o3d
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

from dataset.downloader import ZipUrlDownloader
from logger import logger
from pathlib import Path
import shutil

"""
CO3D Dataset for 3D Object Reconstruction from Multi-view Images.
CDN links file: https://scontent.fprg4-1.fna.fbcdn.net/m1/v/t6/An_tlCbE1hnVIBR2LJJWNbGINO9Jj5_Rmu9KGNdrDm_PoQ4xY3WuRbDIIfdKeiiBcgb8vJ0.txt?_nc_gid=_d85L9YB6LLgoNLOCemmvw&_nc_oc=AdnuRcHknvtGhCEr9BSmg0XDRAQdMnpI9ob_IAmQAot-pDcFdasaphjwUX4aS_F66y8&ccb=10-5&oh=00_AfvHDBl1hyX89Lc1NzZqXOTL5jCV1K6F8SkOT2q4cZtzxg&oe=69A81234&_nc_sid=ba4296
"""


class CO3DDataset(InMemoryDataset):

    category_ids = {
        "apple": "https://scontent.xx.fbcdn.net/m1/v/t6/An_QSC6hT-8cb3Gd3PJ9U2VYscdbDWFj_Ny11wi4ptmIspsE70S_BTc8R6OkSBdIZzWNbqbOu6LEyWovGIk.zip?_nc_gid&ccb=10-5&oh=00_AftQyUE8X9HSlKxkm25TzwKFR54UtbFLgOLdFaXoTHD4Kw&oe=69A7CC36&_nc_sid=ba4296",
        "backpack": "https://scontent.xx.fbcdn.net/m1/v/t6/An9tyyq7fEyndpQjdl4d2UbqMuyGGEYRt32qZRLnxTLLpCQ2PC1QzurnpThigtMS9iS8ggbGL67p7t4aKqg.zip?_nc_gid&ccb=10-5&oh=00_AftYccf0TTFTGEsh9VTvQRKdpXjy1SOmdOttnNJapwq3ww&oe=69A7E680&_nc_sid=ba4296",
        "ball": "https://scontent.xx.fbcdn.net/m1/v/t6/An-jG2tlOIue1umcM2xNXEiGJ89IUpNYdfoa5RTWpBUzaDyS_ZyyUCwEznmuQ6K_cN6THR-IpOCSTXzN6QQ.zip?_nc_gid&ccb=10-5&oh=00_AfsNOegHSRtnMFuSSW7KsjOeWhvISulBQBPTmQiYSvof-A&oe=69A7BDC6&_nc_sid=ba4296",
        "banana": "https://scontent.xx.fbcdn.net/m1/v/t6/An9eGc31j9mtSwJJynbI8cCiLecCcwGQ2Q7V5Q9aTcWxHQQqNFbR9LeIL2WmC12AVvzJGPDIHSeRwJVCC1U.zip?_nc_gid&ccb=10-5&oh=00_AfuCdx7m9Ccx3QMRgzbU_9Xq3PKhlDmah84U9pmO63uYrg&oe=69A7AFDD&_nc_sid=ba4296",
        "baseballbat": "https://scontent.xx.fbcdn.net/m1/v/t6/An9Hwjr4vTebobXoMz51rKFcl2ucITXLOkYW3_Bj0T5eyiH00u80KE0U2e5WpMFAz49CWFl6qFQKhJ0pt2k.zip?_nc_gid&ccb=10-5&oh=00_Afu73Wjddkjq3WIHBGly_x6qzTdfCpMSvWqj1qGmsLG-NQ&oe=69A7AF24&_nc_sid=ba4296",
        "baseballglove": "https://scontent.xx.fbcdn.net/m1/v/t6/An-RX8sWNUppkI_aRRTS4KzBKNmojiH4yf_Zo_-KpSamp_Br6NiIQrkeEuVQy0qV8qHrfW4NY8GJd0fCXPg.zip?_nc_gid&ccb=10-5&oh=00_Aft_8UnKHFsmJZJQgUVTOXMdYDt0CfLqqqOqo36o9PnC8g&oe=69A7CC7F&_nc_sid=ba4296",
        "bench": "https://scontent.xx.fbcdn.net/m1/v/t6/An9NIw_tDVj0SM18igW5FfiYlyRVUWGdLK5mh8qguq_JDZ4yqyMElyzRpQw1B_Fto0xnSgzMKcmiNw4IH4w.zip?_nc_gid&ccb=10-5&oh=00_AftUirxqq2xUs5kdKmd_Z_wekCbMhHuFfw6E0O3xLyl_kg&oe=69A7DE72&_nc_sid=ba4296",
        "bicycle": "https://scontent.xx.fbcdn.net/m1/v/t6/An_Rm4gpdXvhAhCBYSw6j3LYr8KBiNff_qhJJiIJOBVXpjLki0Ezu5DmB1LITPP3hjlzE5g4qBacdpMNkQw.zip?_nc_gid&ccb=10-5&oh=00_AfuiYjFiTZZscpZG_tv1YrinUSMOuDZqYB5r7bJ1fz8Kmw&oe=69A7B59E&_nc_sid=ba4296",
        "book": "https://scontent.xx.fbcdn.net/m1/v/t6/An9W1H-8Vz1T584S6E56WuY4M1NAjrnFDC0Qi3ERfIUq0OzDQEv-vYX8t7WX6wP8IIQ_oHeJRYiwFtwTLUM.zip?_nc_gid&ccb=10-5&oh=00_AfsAGolxTPOhSrfAxkjSL3gZgq0upjsMKK9qqRopjVwllg&oe=69A7B123&_nc_sid=ba4296",
        "bottle": "https://scontent.xx.fbcdn.net/m1/v/t6/An_UFwIvyyap4rLtrzCzZB3GFpJb6vd3rocOdyXKxOWVl5AKXAoVbCe5Vs2Z6P_63vYKKt3ji_VsMX6--fc.zip?_nc_gid&ccb=10-5&oh=00_Aft_AnbYLfEC3KAtabsLrV63mupFqIY2Q9FWf8Hqt0r_Ag&oe=69A7CD38&_nc_sid=ba4296",
        "bowl": "https://scontent.xx.fbcdn.net/m1/v/t6/An9nC5fDvFb2P-r_7k3gCEqrwpNqSL48WBY2_1Zt_10s69DBaB5DaYGLWtqa0nDn-ygxsU1Z_1KLS57sQsY.zip?_nc_gid&ccb=10-5&oh=00_Afu_W0_dDAQUgdCNmp8MiyiVF8v0TXdS9f3tD_tKoqtqnA&oe=69A7E395&_nc_sid=ba4296",
        "broccoli": "https://scontent.xx.fbcdn.net/m1/v/t6/An__h7NUFI_CPHV_vSVVsCQPGvhbAfBnUEQEkNNAi55oHTG5IwQVGCTT5skOAgOF1X_Ez7BGCuW-G2Sn9cQ.zip?_nc_gid&ccb=10-5&oh=00_AfucH45bQfFSh61aDuuF39Y-Q9yA8Dn2xDkYsXZgD6dQXQ&oe=69A7BD7F&_nc_sid=ba4296",
        "cake": "https://scontent.xx.fbcdn.net/m1/v/t6/An9GAQcwtTg8srZoI4t1VfIuErH-s_Hhw5_kytqBBTk4hlnat_OH7Ei5ayVGoZVLxO69nR0MLUsKoPhM2fM.zip?_nc_gid&ccb=10-5&oh=00_AfuShKP6kKBKSkQMAJ01qvZWesGomp8P8ycpqmLVbMkzjA&oe=69A7C86F&_nc_sid=ba4296",
        "car": "https://scontent.xx.fbcdn.net/m1/v/t6/An-fxZJgr0X0iIWhTjt-LD_MqBxf0SVGno1ggNYwEkB9zq6nMuQaysGo2nO_T5hvoRX_gDkTEBqSD7GuyGY.zip?_nc_gid&ccb=10-5&oh=00_Afu2YavIg-8NKnaIPa9KZ9WRu56nC8X32Udlc5C6ASc5lw&oe=69A7E06F&_nc_sid=ba4296",
        "carrot": "https://scontent.xx.fbcdn.net/m1/v/t6/An_U98H5kASgvgePfSCU4dt_uMwydiNuX8AX6mXWZpkEPt85PtzMhNvNzFbuk2L9sszj07GglvTXAtUTHe4.zip?_nc_gid&ccb=10-5&oh=00_AftIiYT6M3xLCcrxZnl_pBQYvq0k3vC-DKVjW6B0ItgvTw&oe=69A7E5E2&_nc_sid=ba4296",
        "cellphone": "https://scontent.xx.fbcdn.net/m1/v/t6/An_rwJqQRlwsSV509Iy6uzJuinXA-R78DnrrSl_H4ufdgom5X3E8KLut7xXh-gqOVD-VzomVsOqRbq5A9N0.zip?_nc_gid&ccb=10-5&oh=00_AftsbjFzuhGywl3H_SoeYALVU-32oS2AZD4jgQZe6U4S2A&oe=69A7C31A&_nc_sid=ba4296",
        "chair": "https://scontent.xx.fbcdn.net/m1/v/t6/An-kqzDg5pX2UxZbhRA4Hd4d8yWaMzG8C9FauxeYN7jQOz5Tuhg5znQKpVo0VCDnFD7Y1XnnPrZXXySNofc.zip?_nc_gid&ccb=10-5&oh=00_Afv5RlE21jjijy7-4B3ZjwuMCVSxZrCyDrXlCMjUJtLybQ&oe=69A7D73E&_nc_sid=ba4296",
        "couch": "https://scontent.xx.fbcdn.net/m1/v/t6/An-Y531FBt1_Dmjmdml0fAxWVznPEZ4KhVulHN4RqyLIqKI4Fldv1Q2EBkOSRtG8co5l0O9EVtYm894u1-w.zip?_nc_gid&ccb=10-5&oh=00_AfsXTPS38wfm25gqL39fkGNpnIgXZYoHLH0iiwCc6_qj5A&oe=69A7B747&_nc_sid=ba4296",
        "cup": "https://scontent.xx.fbcdn.net/m1/v/t6/An_IG6IFIimI4F3KOJrIJt8loZF8iELYHDcBrNBxp686y8YTuPeet6hQ_os5K0uI3GnbXQRinE2Y9-304BU.zip?_nc_gid&ccb=10-5&oh=00_AfuHUzL3ps8cdZ2j-Pwqw6ThrYpW6A4e5CXA736Fr9RUiA&oe=69A7DA09&_nc_sid=ba4296",
        "donut": "https://scontent.xx.fbcdn.net/m1/v/t6/An8LItMvwmV6Mg5ucjsR8J8ZAW6dNRHzKs3AS6wTX9dhCAPJtPRdxs0E1itjQENEsp404WVyfPJJqW_W2cw.zip?_nc_gid&ccb=10-5&oh=00_Afu7ubxxrPzkxTlXWH0SgSpVOwp2RqR7JARtnOyEaLO4bg&oe=69A7BD01&_nc_sid=ba4296",
        "frisbee": "https://scontent.xx.fbcdn.net/m1/v/t6/An_0OjAgLVTKieSvSpVQTGPSAb0oqhPRob74uSK2w2lZFSS5i-kTUoTWyc9arl2e4DndFad3qHv8CBDPMFo.zip?_nc_gid&ccb=10-5&oh=00_Aft9iKfW54W01VaBiSdc8Y2yrolIA3-ZUU87D2j5OeeYXQ&oe=69A7B733&_nc_sid=ba4296",
        "hairdryer": "https://scontent.xx.fbcdn.net/m1/v/t6/An94gPdry5iLgaEbyUt_0EMyFrerhQpSGOPK564nTarWoVwHB2ZVzreuAdEZPniYfnU7sR4NCKVisuiBFpw.zip?_nc_gid&ccb=10-5&oh=00_Afur50vK6m0CbfvuRzaFoeBBM4cXOXdY8lh_kd5aE9IR1A&oe=69A7D737&_nc_sid=ba4296",
        "handbag": "https://scontent.xx.fbcdn.net/m1/v/t6/An8n_p4JlXoz4XXPX0A3JxxYgpQAs7ALq3bwlQEDERlyaDeebtZUq4TiwasrSTx5atKUnQKAOQv46dG0jsM.zip?_nc_gid&ccb=10-5&oh=00_AfuoQaLyzcPyyC1yS29k7RCOnnMKwh7E1hlXaPLmupDEIA&oe=69A7E6B7&_nc_sid=ba4296",
        "hotdog": "https://scontent.xx.fbcdn.net/m1/v/t6/An9OQMTav9nQ7B9Jhd_H2vq4UF7hvfdHPfVRZltSWqlU-sh9tZgYT_MaWn0-3u9RrgdUnyuKhx_eolT7-W0.zip?_nc_gid&ccb=10-5&oh=00_AftX0EPrpG5BhhpEb6Js1TegoKtk4MBf2wMl_fcGSNqwuw&oe=69A7CA7B&_nc_sid=ba4296",
        "hydrant": "https://scontent.xx.fbcdn.net/m1/v/t6/An_WQ2hli3BcR5TKb_Gr0A9GsvgVUf8eMwFfWWwKnj2zj1bWAZWHKcp3OfaSZR9gfoYQaEeKV4EpopBLXuk.zip?_nc_gid&ccb=10-5&oh=00_AfsxB_gxjPBi_6kB_EGGUdGM8_EvgfDL2nOZBYQ3VdVAew&oe=69A7B1C5&_nc_sid=ba4296",
        "keyboard": "https://scontent.xx.fbcdn.net/m1/v/t6/An-oNTCjkPBLtTVYK4qqCun0X6JgPf0su69XKzhbM2Zoks0usg3XY1JD2ukBO-P6uyR0zHrYuVNkd6tS2Ps.zip?_nc_gid&ccb=10-5&oh=00_Afv2ov-IvVngBxOD0W3gL3sb9_dkB2yYTdVHY_aUfFa9Bg&oe=69A7BBDA&_nc_sid=ba4296",
        "kite": "https://scontent.xx.fbcdn.net/m1/v/t6/An8H26CqbMq5HPuB7C_AIlw6uloZ1N7azpOj6haqJnwgTkGaJBDgFg7siBChwvMVzxjC7oi_UEPFX_2nBqc.zip?_nc_gid&ccb=10-5&oh=00_AfsF5Mss0XlL9dwM2EYiKdJ-jMWh4CaLzZSnLIB1HPgYTA&oe=69A7D0B6&_nc_sid=ba4296",
        "laptop": "https://scontent.xx.fbcdn.net/m1/v/t6/An9c0OYTwmu5xc9JhqLv5vjnuIRiJnjlG1AS9fo37Smsusw_zuq-fujnYo3M8Tok4DUzhQXV8IRyRYJOh_k.zip?_nc_gid&ccb=10-5&oh=00_Afv2JPkUpCn7JO6ziaJwDk-l-dEb7opL0XrymwA_OWkQ2A&oe=69A7E307&_nc_sid=ba4296",
        "microwave": "https://scontent.xx.fbcdn.net/m1/v/t6/An_T1fkClfpPU0DOxdE94ybPsjfI8Nh7KaagsI3IRFojgXlSRt4tBQCHxAcHfEkXuZ2cbvPVZ9RNCbc9B-s.zip?_nc_gid&ccb=10-5&oh=00_AfvahhlmDKAWdud5Y3X6UHRb-vkWfbC1EAyuUrbedqoxfQ&oe=69A7B0D0&_nc_sid=ba4296",
        "motorcycle": "https://scontent.xx.fbcdn.net/m1/v/t6/An8OosNfooF0_fEse8t8aN56WkxkD8eAGyKLR3JNHgRNfnTARECZqDsOGlNgpEKvF1vzEY2h1zZLVjtqKrs.zip?_nc_gid&ccb=10-5&oh=00_AfsGOFfUReCJ_zU709Cp99WIZMTXmtV6EXxjaeV-dMVs0g&oe=69A7E0DD&_nc_sid=ba4296",
        "mouse": "https://scontent.xx.fbcdn.net/m1/v/t6/An95CAZ2t5EEZaSNq3Cr57NKsYaLCt7y1a_9WHwi0bVobKX9XB4vFGEZLoAO3x5AnrAn6nzG7RsiL1_WFTo.zip?_nc_gid&ccb=10-5&oh=00_Afui3yLnYkdWWPrFMu6C-VUQlJsLSzrxV3rzOHrJqE32gA&oe=69A7D794&_nc_sid=ba4296",
        "orange": "https://scontent.xx.fbcdn.net/m1/v/t6/An8aBfGHNIVQPaaQGHzkzWnEsucc42wu-Ban-VnuMYkylpvQuK-yNA8_EPfN9qDktEDcBz03yb1QFR73Sas.zip?_nc_gid&ccb=10-5&oh=00_Aftu-akiG5lUfssc9ZvRwvDVHw-0AAhIwzU4RRqi9poB0Q&oe=69A7E257&_nc_sid=ba4296",
        "parkingmeter": "https://scontent.xx.fbcdn.net/m1/v/t6/An_O7b6axtJW94I7Jgb0VFZQo21zwFwZMSA00uxJiCBpTSP2HnU9spp_zQnqZL8cE-FKLLhuWT8MIr5pPcU.zip?_nc_gid&ccb=10-5&oh=00_AfvLqFfQzTU48zBfq5FnC1DqbCHZM-6KEDPkNv7DeLhppQ&oe=69A7CC21&_nc_sid=ba4296",
        "pizza": "https://scontent.xx.fbcdn.net/m1/v/t6/An8CvVz2_o4RPylyeqL3t12hUM9H1zLrk4tK6uv5c3uAO5NIlyNttgVU5iIhwZBL77D-GwHSxcpyOL4NHgE.zip?_nc_gid&ccb=10-5&oh=00_AftsVTcgajSDOdKKQDyQ0E8JLK7DbXV4U_dD_P9Nm9aiog&oe=69A7E3E7&_nc_sid=ba4296",
        "plant": "https://scontent.xx.fbcdn.net/m1/v/t6/An8ImEtYSyLU5yCtxIUWI7hejNz51b_NChyCNN5OwFl_V7JCHu41Z1rshq75maZ8nMeaviIK3HooWt0ttEc.zip?_nc_gid&ccb=10-5&oh=00_AfuKIYORJTyYpUcDt2Hd1l_VIcXAGg9LewGZDAAWl4MgxQ&oe=69A7B62A&_nc_sid=ba4296",
        "remote": "https://scontent.xx.fbcdn.net/m1/v/t6/An8qolzrQcDTE_3q2tGM8uEclxVHpqeOmxRCKydENMljfX0PXoT4LSKQ0bfUhS9sGXzl7tzP1MgN9wMIcXU.zip?_nc_gid&ccb=10-5&oh=00_Afunq-4WhKlrlcKw1DGrRY3zyjWthUQf-Nkim85f5QrNyQ&oe=69A7C095&_nc_sid=ba4296",
        "sandwich": "https://scontent.xx.fbcdn.net/m1/v/t6/An-dmeRk_G2_7mJ3R-TrZvzf9FjvpProxE0hL1CMAA2jc1GW03eOoqaue__zQDF-pw72DCDfhD-4P72AfK0.zip?_nc_gid&ccb=10-5&oh=00_AfvIr_xM30NObK9rlNHBe49bKb4kc4x9CycCBVTB_mItaA&oe=69A7AF20&_nc_sid=ba4296",
        "skateboard": "https://scontent.xx.fbcdn.net/m1/v/t6/An8pE5f4i4z6W5Fk1JK_ZtpL2CDEG7siJetk1YDAinUXlY9zNze6Sv4Gj2lLD8noQKQNXs8jUXwROdZAmr4.zip?_nc_gid&ccb=10-5&oh=00_Afsomhl5Z--6UzF9pcBonjUjOtwrXmd_hB_EEYmCqE0jOQ&oe=69A7BE52&_nc_sid=ba4296",
        "stopsign": "https://scontent.xx.fbcdn.net/m1/v/t6/An-tR_-d_PK4GDMQBVzhzzazt9ONP4py_TZoy80h4Hea9TPO55fWaL9oHqM72cDfof1zTa-Xkey0sEp5B7M.zip?_nc_gid&ccb=10-5&oh=00_AfskyY1x2A-2uUDG-CP2eSamCZWnTj2k6ThYtznpM5SNbQ&oe=69A7D2EE&_nc_sid=ba4296",
        "suitcase": "https://scontent.xx.fbcdn.net/m1/v/t6/An8x0EmZ3Fq_ElbH9O3kno69pwCHHnwCW1nErqgsMAsNv4QQSzV5Naif2hQ6fjiHmXtX7xx1jZdUBLBUwpc.zip?_nc_gid&ccb=10-5&oh=00_AfuJKs_eTeXnVwiNLNrW1MGnlbmrb_KUnG32GvFwT8qVfg&oe=69A7D79C&_nc_sid=ba4296",
        "teddybear": "https://scontent.xx.fbcdn.net/m1/v/t6/An9J9NR9EKKW9kr9CH47myJVPREBeLdttjyQYBhSbr3pkRIiox47R9I6swgjg6Yb-6L3AZ4-2LgxVhsjJDc.zip?_nc_gid&ccb=10-5&oh=00_AfvObJigoMM2cgnHb272Q8kqN6o9Mg5M2QINn_N8rPupUA&oe=69A7CE7B&_nc_sid=ba4296",
        "toaster": "https://scontent.xx.fbcdn.net/m1/v/t6/An8RczSLJSc71Gg28i5tR4ivaM7MFoKM6fnKe6btpf9tGpMI8IkUEVGRJ12-bmKZh4heFr8MAjSt2WxFtIg.zip?_nc_gid&ccb=10-5&oh=00_AfsqlY8nWIuFVSUOYxQUq0-Xr65iYaVD58L7iXPrl0nXJw&oe=69A7C8E1&_nc_sid=ba4296",
        "toilet": "https://scontent.xx.fbcdn.net/m1/v/t6/An-BtdPVEYTkaZz_lKz5eQflO84MYFVkrsexLySr0wiw8CBnv_Xcmmnm8hzOsUKMlRqRlbesiPXEEVpKTJE.zip?_nc_gid&ccb=10-5&oh=00_AfvAz8M3_1VGXo4EvkyjA8FeMCsKvKUI9lRMxg2J3353sg&oe=69A7E370&_nc_sid=ba4296",
        "toybus": "https://scontent.xx.fbcdn.net/m1/v/t6/An-e2Mta1zEnPsoQhm2M63T9oADPlecMO8iP8F3s8FBdQDItNZR-djYWoVXDvle7AbVK0pES_xNJlkNcOqo.zip?_nc_gid&ccb=10-5&oh=00_AfusvnG8XClpqehWSfPjdjojq8s8UciVKiwZAAlF5Yp_8A&oe=69A7AFBF&_nc_sid=ba4296",
        "toyplane": "https://scontent.xx.fbcdn.net/m1/v/t6/An-nluyYsQNpA5H6qo7bDSHX4mtpmie0CCQHnPq7-asIpr6p28VDmHWSekAis6tSNPGaiI2Dx7wl7E6D8CQ.zip?_nc_gid&ccb=10-5&oh=00_AftRRLXsD6vFl5xxdK44_W-S2YsNuBf1DDrduTOxowdPyg&oe=69A7BEC9&_nc_sid=ba4296",
        "toytrain": "https://scontent.xx.fbcdn.net/m1/v/t6/An_GnWcAoqhXO6-JvJzgMzsy_ZN1pMZaUlY-K9Yvq95GZDQRZNkXNrco271lErLDAX6IqLexmLNSOjcJAdE.zip?_nc_gid&ccb=10-5&oh=00_AftcNYzbf0CiwEAKZvCD5hpnPFmhMm5cOE2zUvKwLuHcQg&oe=69A7B385&_nc_sid=ba4296",
        "toytruck": "https://scontent.xx.fbcdn.net/m1/v/t6/An_W7VsuDvxZUS0Ms7gMEEhxTJjg6wlrMWIGA2BfNqpTQpSq7VbKAF5eriJQ43_9uG9KqztuYNqyAMwo2FI.zip?_nc_gid&ccb=10-5&oh=00_AftFi_tGisLOrJUd9qiyzxrdUeqiI0TliwgSB2ddSOdBEQ&oe=69A7C7E8&_nc_sid=ba4296",
        "tv": "https://scontent.xx.fbcdn.net/m1/v/t6/An_Dwy0JQM-ff-imtBd_tm1ysnkYvnLVobheO_amBsmzoeU_bmH3l9NQ9F3on4YArNSWXWpccpeIvNJg3iI.zip?_nc_gid&ccb=10-5&oh=00_Afu13VVmdVSan_5n2W_qij0f69K_9kecuQwgF8Exk7SG3A&oe=69A7D7CA&_nc_sid=ba4296",
        "umbrella": "https://scontent.xx.fbcdn.net/m1/v/t6/An_fbQlYbVAgMOFjTOY6Evj7IJ2VO6WmCvdX4pqIaHGGkXnJlFDzbg_Il7B_UbrlCwdrYcO2fmIDOnK0jsU.zip?_nc_gid&ccb=10-5&oh=00_Afsd4UyJ8-cqWkYKqxX5kTIFMAK2IPbigcZvGbb8z7-ABw&oe=69A7BF32&_nc_sid=ba4296",
        "vase": "https://scontent.xx.fbcdn.net/m1/v/t6/An9qdS5KidE_caUH0nPX5StCH_u9Xwt2wH3xU6om0p6ZaYK3JAja80iMRG2LHsKCZkm1ul8YoA4KG3LuqWo.zip?_nc_gid&ccb=10-5&oh=00_AfsBRijMVAMw4ecaqg9Z5zZi3QqXpS3lovWzK4Ac3-9ZLw&oe=69A7C52C&_nc_sid=ba4296",
        "wineglass": "https://scontent.xx.fbcdn.net/m1/v/t6/An-D7nh5JqEI-3bEtEfyAdCmryr3Zc1mQsd_sFxuIQ6g1E_sYuDerfwJB7j7ZBGa7Wa-I_Uzn3yzaaNPbQ8.zip?_nc_gid&ccb=10-5&oh=00_Afvy-KbW1jUCITIoMOU5PJNgHYu9Ub-sWGCvvcmcKdJHqw&oe=69A7E645&_nc_sid=ba4296",
    }

    def __init__(
        self,
        root: str,
        categories: Optional[Union[str, List[str]]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.root = root

        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]

        self.categories = categories
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super().__init__(root, log=False)

        if force_reload:
            logger.info("Force reload requested; clearing processed directory.")
            self._remove_processed()

        # If processed/ is empty → process automatically
        if len(self.processed_file_names) == 0:
            logger.info("Processed dataset not found. Processing dataset now...")
            self.process()

        # Build final file list
        self.files = sorted(os.listdir(self.processed_dir))

    @property
    def raw_dir(self):
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return list(self.category_ids.keys())

    @property
    def processed_file_names(self) -> List[str]:
        if not osp.exists(self.processed_dir):
            return []
        return [f for f in os.listdir(self.processed_dir) if f.endswith(".pt")]

    def download_postprocess(self, extract_dir: Path):
        """
        Postprocessing callback to clean up CO3D dataset structure.
        Keeps only 'masks' and 'images' folders from the nested tv directory,
        removing all other files and folders.
        """
        # Find the first folder inside extract_dir
        first_folder = None
        for item in extract_dir.iterdir():
            if item.is_dir():
                first_folder = item
                break

        # Move content of first folder to extract_dir
        if first_folder:
            # Iterate hierarchically and move only 'masks' and 'images' folders
            def process_directory(directory: Path):
                for subitem in directory.iterdir():
                    if subitem.is_dir():
                        # Check if this directory contains masks or images folders
                        has_masks_or_images = any(
                            child.is_dir() and child.name in ["masks", "images"]
                            for child in subitem.iterdir()
                        )

                        if has_masks_or_images:
                            # This is a parent folder containing masks/images
                            dest_path = extract_dir / subitem.name
                            if dest_path.exists():
                                shutil.rmtree(dest_path)

                            # Copy the entire parent folder
                            shutil.copytree(str(subitem), str(dest_path))

                            # Clean up the copied folder to keep only masks and images
                            for copied_item in dest_path.iterdir():
                                if copied_item.is_dir() and copied_item.name not in [
                                    "masks",
                                    "images",
                                ]:
                                    shutil.rmtree(copied_item)
                                elif copied_item.is_file():
                                    copied_item.unlink()

                            # Remove the original directory
                            shutil.rmtree(subitem)
                        else:
                            # Recursively process subdirectories
                            process_directory(subitem)
                            # Remove the directory after processing
                            if subitem.exists():
                                shutil.rmtree(subitem)
                    else:
                        # Remove files
                        subitem.unlink()

            process_directory(first_folder)

            if first_folder.exists():
                first_folder.rmdir()

    def download(self):
        for category in self.categories:
            downloader = ZipUrlDownloader(
                local_dir=osp.join(self.raw_dir, category),
                remote_dir=self.category_ids[category],
                postprocess_callback=self.download_postprocess,
            )

            downloader.download()

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = osp.join(self.processed_dir, self.files[idx])
        data = torch.load(file_path, weights_only=False)

        if self.transform:
            data = self.transform(data)
        return data

    def _remove_processed(self):
        if osp.exists(self.processed_dir):
            for f in os.listdir(self.processed_dir):
                os.remove(osp.join(self.processed_dir, f))

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        logger.info(f"Processing categories: {self.categories}")
