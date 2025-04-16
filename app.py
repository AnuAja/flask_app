from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import joblib  # Untuk memuat model yang sudah dilatih

app = Flask(__name__)

# Load model KNN yang sudah dilatih
model = joblib.load('knn_model.pkl')

# List gejala sesuai dataset
list_gejala = [
    "Lepuhan di mulut",
    "Luka di kuku",
    "Demam",
    "Air liur berlebih",
    "Nafsu makan menurun",
    "Pincang",
    "Produksi susu menurun",
    "Keguguran",
    "Kelenjar susu bengkak dan memerah",
    "Pembengkakan kelenjar susu",
    "Pembengkakan sendi",
    "Pembengkakan testis",
    "Kelahiran anak sapi lemah",
    "Kematian mendadak",
    "Keluarnya darah hitam dari lubang tubuh",
    "Sesak napas",
    "Kejang",
    "Keluar cairan dari mata dan hidung",
    "Keluar darah dari mulut atau hidung",
    "Hidung berlendir",
    "Mata berair dan belekan",
    "Susu berwarna keruh atau bercampur darah",
    "Nyeri saat pemerahan",
    "Penurunan berat badan",
    "Nafas berbau manis",
    "Lesu",
    "Anemia",
    "Benjolan di kulit sebesar kelereng",
    "Pembengkakan di kaki",
    "Bau busuk dari kaki",
    "Kesulitan berdiri atau bergerak",
    "Pembengkakan leher",
    "Gangguan pernapasan",
    "Gatal-gatal pada kulit",
    "Pembengkakan di sekitar mata",
    "Mata tertutup",
    "Bintik putih di mata",
    "Mata keruh",
    "Diare",
    "Bulu kusam",
    "Telinga terkulai",
    "Kekurusan",
    "Kerak-kerak pada kulit",
    "Kerontokan bulu",
    "Kulit menjadi tebal dan kaku",
    "Kulit kemerahan atau mengering",
    "Kulit melepuh",
    "Lesi di mulut",
    "Feses berbau menyengat",
    "Feses disertai bercak darah"
]

deskripsi_penyakit = {
    "Penyakit Mulut dan Kuku (PMK)": {
        "deskripsi": "Penyakit mulut dan kuku (PMK) adalah penyakit menular yang menyerang hewan berkuku belah, seperti sapi, kerbau, domba, kambing, dan babi.",
        "gejala": ["Lepuhan di mulut", 
                    "Luka di kuku",
                    "Demam",
                    "Air liur berlebih",
                    "Nafsu makan menurun",
                    "Pincang sampai tidak dapat berdiri",
                    "Produksi susu menurun pada sapi perah",
                    "Keluron / abortus pada sapi hamil"
                    ],
        "penanganan": ["Pisahkan hewan yang sakit dengan yang sehat pada kandang terpisah.",
                        "Semprotkan disinfektan di lingkungan hewan.",
                        "Jangan pegang hewan yang sakit, dan jika memegang, segera mandi setelahnya.",
                        "Jangan berikan sisa makanan hewan yang sakit kepada hewan yang sehat.",
                        "Jangan mencampur peralatan hewan yang sakit dengan yang sehat."
                        ]
    },
    "Brucellosis (Keluron Menular)": {
        "deskripsi": "Brucellosis adalah penyakit infeksi yang disebabkan oleh bakteri <i>Brucella abortus</i>. Penyakit ini dapat menular dari hewan ke manusia, sehingga disebut juga penyakit zoonosis.",
        "gejala": ["Keguguran di umur kehamilan 6 - 9 bulan",
                    "Pembengkakan kelenjar susu",
                    "Demam",
                    "Pembengkakan sendi ( Sering terjadi pada sapi Jantan )",
                    "Pembengkakan testis ( Pada sapi Jantan )",
                    "Produksi susu menurun pada sapi perah",
                    "Kelahiran anak sapi lemah sampai dengan mati"
                    ],
        "penularan": ["Mengonsumsi daging yang kurang matang, seperti daging sapi, kambing, domba, atau unta.",
                        "Mengonsumsi susu mentah atau produk susu lainnya yang tercemar dan yang tidak dipasteurisasi.",
                        "Kontak langsung dengan hewan yang terinfeksi, peralatan serta sekreta (hasil ikutan yang keluar dari tubuh hewan terinfeksi."
                        ]
    },
    "Anthrax (Radang Limpa)": {
        "deskripsi": "Antraks adalah penyakit infeksi bakteri yang dapat menular dari hewan ke manusia (zoonosis), Antraks dapat menyerang berbagai hewan berdarah panas, termasuk sapi, kambing, domba, kuda, dan babi. Penyakit ini disebabkan oleh bakteri <i>Bacillus anthracis</i> yang hidup di tanah.",
        "gejala": ["Demam",
                    "Kematian mendadak",
                    "Keluarnya darah hitam dari lubang tubuh ( bisa iya / tidak )",
                    "Nafsu makan menurun",
                    "Sesak napas ( bisa iya / tidak )",
                    "Kejang ( bisa iya / tidak )"
                    ],
        "penularan" : ["Kontak langsung dengan hewan atau produk hewan serta peralatan ( ember, sekop, dll ) dan kandang yang terkontaminasi."
                        "Terhirupnya spora antraks ke saluran pernapasan."
                        "Mengonsumsi daging yang kurang matang, seperti daging sapi, kambing, domba, kuda atau babi."
                        ]
    },
    "Bovine Viral Diarrhea (BVD)": {
        "deskripsi": "BVD atau Diare Virus Sapi adalah penyakit infeksius pada sapi yang disebabkan oleh virus <i>BVDV</i>. Penularan BVD bisa secara kontak langsung dengan hewan terinfeksi dan tidak langsung melalui makanan yang terkontaminasi feses.",
        "gejala": ["Keluarnya cairan dari mata dan hidung",
                    "Lesi di mulut",
                    "Lesu",
                    "Demam",
                    "Nafsu makan menurun",
                    "Diare",
                    "Produksi susu menurun"
                    ],
        "penanganan": ["Pisahkan hewan yang sakit dengan yang sehat pada kandang terpisah.",
                        "Melakukan vaksinasi yang dilakukan tenaga kesehatan hewan."
                        ]
    },
    "Mastitis (Radang Ambing / Kelenjar Susu)": {
        "deskripsi": "Mastitis adalah radang pada kelenjar susu sapi, biasanya disebabkan oleh bakteri seperti <i>Staphylococcus aureus</i> dan <i>Escherichia coli</i>. Mastitis dapat dikenali dengan mudah jika ambing atau kelenjar susu mengalami 3A ( abang, aboh, anget ).",
        "gejala": ["Demam",
                    "Kelenjar susu membengkak dan memerah",
                    "Produksi susu menurun bahkan sampai susu tidak keluar",
                    "Susu berwarna keruh atau bercampur darah",
                    "Nyeri saat pemerahan"
                    ]
    },
    "Ketosis": {
        "deskripsi": "Ketosis terjadi akibat gangguan metabolisme ketika sapi kekurangan energi, terutama setelah melahirkan.",
        "gejala": ["Nafsu makan menurun",
                    "Penurunan berat badan",
                    "Nafas berbau manis",
                    "Produksi susu menurun"
                    ],
        "penyebab": ["Ketidakseimbangan ransum, seperti kekurangan karbohidrat atau kelebihan lemak",
                    "Stres kondisi, seperti transportasi, kepadatan penduduk, dan kondisi cuaca ekstrem",
                    "Genetik, beberapa hewan lebih rentan terhadap ketosis",
                    "Ketidakseimbangan hormon, seperti hipotiroidisme atau gangguan metabolisme lainnya"
                    ]
    },
    "Scabies (Kudis)": {
        "deskripsi": "Scabies pada sapi adalah penyakit kulit yang disebabkan oleh tungau <i>Sarcoptes scabiei bovis</i>. Penyakit ini juga dikenal sebagai kudis sarkoptik atau kudis pada sapi. Penyakit ini sering menyerang ternak yang kekurangan pakan, di musim kemarau, dan di lingkungan kandang yang kotor.",
        "gejala": ["Gatal-gatal pada kulit",
                    "Kerak-kerak pada kulit",
                    "Kerontokan bulu",
                    "Kulit menjadi tebal dan kaku",
                    "Kulit kemerahan atau mengering",
                    "Kulit melepuh, terutama di daerah muka dan punggung"
                    ],
        "penularan" : ["Kontak langsung dengan hewan yang terinfeksi.",
                        "Melalui benda yang terkontaminasi."
                        ]
    },
    "Lumpy Skin Disease (LSD) (Lato-lato)": {
        "deskripsi": "LSD disebabkan oleh <i>Lumpy Skin Disease Virus (LSDV)</i>, yang termasuk dalam keluarga <i>Poxvirus</i>.",
        "gejala": ["Benjolan di kulit sebesar kelereng ( diameter 2 – 3 cm ) terlihat utuh atau sudah pecah",
                    "Demam",
                    "Penurunan produksi susu tidak signifikan",
                    "Keguguran tidak signifikan",
                    "Nafsu makan menurun",
                    "Kekurusan"
                    ],
        "penanganan": ["Pisahkan ternak yang terinfeksi ke kandang isolasi.",
                        "Jaga kebersihan kandang dan peralatan kendang."
                        ],
        "penularan" : ["Dapat ditularkan melalui gigitan vektor ( lalat, nyamuk, caplak ).",
                        "Penularan secara langsung terjadi melalui kontak dengan lesi kulit."
                        ]
    },
    "Foot Rot (Kaki Busuk)": {
        "deskripsi": "Foot Rot adalah infeksi bakteri <i>(Fusobacterium necrophorum)</i> yang menyerang kuku sapi. Biasanya terjadi di lingkungan yang basah dan berlumpur.",
        "gejala": ["Pembengkakan di kaki",
                    "Demam",
                    "Bau busuk dari kaki", 
                    "Kesulitan berdiri atau bergerak bahkan sampai ndeprok ( tidak mampu berdiri )",
                    "Pincang",
                    "Nafsu makan menurun",
                    "Penurunan produksi susu pada sapi perah"
                    ]
    },
    "Diare Karena Bakteri": {
        "deskripsi": "Diare pada sapi dapat disebabkan oleh berbagai faktor, seperti infeksi bakteri, virus, dan protozoa. Bakteri, seperti <i>E. coli, Salmonella spp.</i> dan <i>Clostridium perfringens</i>.",
        "gejala": ["Diare",
                    "Nafsu makan menurun",
                    "Berat badan menurun",
                    "Lesu",
                    "Feses berbau menyengat",
                    "Feses disertai bercak darah",
                    "Demam"
                    ]
    },
    "Haemorrhagic Septicaemia (Penyakit Ngorok)": {
        "deskripsi": "Penyakit ini disebabkan oleh <i>Pasteurella multocida</i> dan menyerang saluran pernapasan serta sistem peredaran darah.",
        "gejala": ["Demam", 
                    "Pembengkakan leher",
                    "Air liur berlebih",
                    "Hidung berlendir",
                    "Sesak napas",
                    "Gangguan pernapasan",
                    "Keluarnya darah dari mulut atau hidung",
                    "Kematian"
                    ]
    },
    "Pinkeye (Belekan)": {
        "deskripsi": "Pinkeye adalah infeksi mata yang disebabkan oleh <i>Moraxella bovis</i>, menyebabkan peradangan, kemerahan, dan kebutaan sementara atau permanen. Penyakit ini dapat ditularkan oleh lalat yang sering hinggap di muka sapi.",
        "gejala": ["Mata berair dan belekan",
                    "Pembengkakan di sekitar mata",
                    "Mata tertutup karena kotoran mata",
                    "Bintik putih di mata",
                    "Mata keruh",
                    "Lesu"
                    ],
        "penularan" : ["Lalat yang sering hinggap di muka sapi."]
    },
    "Cacingan (Parasit Internal)": {
        "deskripsi": "Cacingan pada sapi adalah penyakit yang disebabkan oleh infeksi parasit cacing. Penyakit ini dapat menyerang sapi dalam berbagai bentuk, seperti cacing gelang <i>(Ascaris lumbricoides),</i> cacing pita <i>(Taenia saginata)</i> dan cacing biji mentimun <i>(Dipylidium sp.)</i>",
        "gejala": ["Penurunan berat badan",
                    "Lesu",
                    "Nafsu makan menurun",
                    "Diare",
                    "Bulu kusam",
                    "Anemia",
                    "Telinga terkulai ( bisa iya / tidak )"
                    ]
    },
    "Tidak Diketahui": {
        "deskripsi": "Penyakit tidak ditemukan berdasarkan gejala yang Anda pilih.",
        "gejala": [],
        "penyebab": [],
        "penanganan": [],
        "penularan" : []
    }
}

@app.route("/")
def home():
    return render_template("persetujuan.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route("/diagnosa", methods=["POST"])
def diagnosa():
    data = request.get_json()
    gejala_terpilih = data.get("gejala", [])

    # Konversi input ke bentuk vektor fitur (0 dan 1)
    input_gejala = [1 if g in gejala_terpilih else 0 for g in list_gejala]
    input_array = np.array(input_gejala).reshape(1, -1)

    # Prediksi penyakit dengan KNN
    hasil_prediksi = model.predict(input_array)[0]  # Ambil hasil prediksi

    # Ambil data penyakit dari dictionary deskripsi_penyakit
    hasil = deskripsi_penyakit.get(hasil_prediksi, {
        "deskripsi": "Deskripsi tidak tersedia.",
        "gejala": [],
        "penyebab": [],
        "penanganan": [],
        "penularan": []
    })

    return jsonify({
        "penyakit": hasil_prediksi,
        "deskripsi": hasil["deskripsi"],
        "gejala": hasil.get("gejala", []),
        "penyebab": hasil.get("penyebab", []),
        "penanganan": hasil.get("penanganan", []),
        "penularan": hasil.get("penularan", [])
    })

@app.route('/hasil')
def hasil():
    data_json = request.args.get("data", "{}")  # Ambil data dari URL
    data = json.loads(data_json)  # Ubah dari string JSON ke dictionary
    return render_template("hasil.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)