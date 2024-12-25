import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import pickle
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def get_abbreviations_dict():
    return {
        # Pemerintahan & Lembaga Negara
        'dpr': 'dewan perwakilan rakyat',
        'mpr': 'majelis permusyawaratan rakyat',
        'dpd': 'dewan perwakilan daerah',
        'kpk': 'komisi pemberantasan korupsi',
        'kpu': 'komisi pemilihan umum',
        'bawaslu': 'badan pengawas pemilihan umum',
        'dprd': 'dewan perwakilan rakyat daerah',
        'tni': 'tentara nasional indonesia',
        'polri': 'kepolisian republik indonesia',
        'pj': 'penjabat',
        'pemkot': 'pemerintah kota',
        'pemkab': 'pemerintah kabupaten',
        'pemprov': 'pemerintah provinsi',
        'pemda': 'pemerintah daerah',
        'asn': 'aparatur sipil negara',
        'dppp': 'dinas pertanian pangan dan perikanan',
        'dishub': 'dinas perhubungan',
        'dinkes': 'dinas kesehatan',
        'disdukcapil': 'dinas kependudukan dan pencatatan sipil',
        'diskominfo': 'dinas komunikasi dan informatika',
        'panrb': 'pendayagunaan aparatur negara dan reformasi birokrasi',
        'banpol': 'bantuan polisi',
        'satpol pp': 'satuan polisi pamong praja',

        # Kementerian
        'kemenkeu': 'kementerian keuangan',
        'kemendagri': 'kementerian dalam negeri',
        'kemenag': 'kementerian agama',
        'kemenhub': 'kementerian perhubungan',
        'kemenkes': 'kementerian kesehatan',
        'kemendikbud': 'kementerian pendidikan dan kebudayaan',
        'kementan': 'kementerian pertanian',
        'kemenperin': 'kementerian perindustrian',
        'kemendag': 'kementerian perdagangan',
        'kemenkumham': 'kementerian hukum dan hak asasi manusia',
        'kemenlu': 'kementerian luar negeri',
        'kemenhan': 'kementerian pertahanan',
        'kemenpar': 'kementerian pariwisata',
        'kemnaker': 'kementerian ketenagakerjaan',
        'kemenpupr': 'kementerian pekerjaan umum dan perumahan rakyat',
        'kemensos': 'kementerian sosial',
        'kemendesa': 'kementerian desa pembangunan daerah tertinggal dan transmigrasi',
        'kemenkominfo': 'kementerian komunikasi dan informatika',
        'kemendikbudristek': 'kementerian pendidikan kebudayaan riset dan teknologi',

        # Badan & Lembaga
        'bi': 'bank indonesia',
        'bps': 'badan pusat statistik',
        'bpjs': 'badan penyelenggara jaminan sosial',
        'bpom': 'badan pengawas obat dan makanan',
        'bnpb': 'badan nasional penanggulangan bencana',
        'bpbd': 'badan penanggulangan bencana daerah',
        'bmkg': 'badan meteorologi klimatologi dan geofisika',
        'bnn': 'badan narkotika nasional',
        'bkn': 'badan kepegawaian negara',
        'bkpm': 'badan koordinasi penanaman modal',
        'bappenas': 'badan perencanaan pembangunan nasional',
        'lipi': 'lembaga ilmu pengetahuan indonesia',
        'brin': 'badan riset dan inovasi nasional',
        'bssn': 'badan siber dan sandi negara',
        'bakn': 'badan administrasi kepegawaian negara',
        'baznas': 'badan amil zakat nasional',

        # Ekonomi & Bisnis
        'apbn': 'anggaran pendapatan dan belanja negara',
        'apbd': 'anggaran pendapatan dan belanja daerah',
        'bumn': 'badan usaha milik negara',
        'bumd': 'badan usaha milik daerah',
        'umkm': 'usaha mikro kecil dan menengah',
        'ump': 'upah minimum provinsi',
        'pdb': 'produk domestik bruto',
        'ppn': 'pajak pertambahan nilai',
        'pph': 'pajak penghasilan',
        'ojk': 'otoritas jasa keuangan',
        'bei': 'bursa efek indonesia',
        'ihsg': 'indeks harga saham gabungan',
        'sbn': 'surat berharga negara',
        'pmk': 'peraturan menteri keuangan',
        'perppu': 'peraturan pemerintah pengganti undang undang',
        'apht': 'akta pemberian hak tanggungan',

        # Teknologi & Digital
        'ai': 'artificial intelligence',
        'iot': 'internet of things',
        'qris': 'quick response code indonesia standard',
        'fintech': 'financial technology',
        'spbe': 'sistem pemerintahan berbasis elektronik',
        'api': 'application programming interface',
        'ml': 'machine learning',
        'cdn': 'content delivery network',
        'dns': 'domain name system',
        'saas': 'software as a service',
        'paas': 'platform as a service',
        'iaas': 'infrastructure as a service',

        # Kesehatan & Sosial
        'covid': 'coronavirus disease',
        'rs': 'rumah sakit',
        'rsud': 'rumah sakit umum daerah',
        'who': 'world health organization',
        'pmi': 'palang merah indonesia',
        'pmk': 'penyakit mulut dan kuku',
        'dbd': 'demam berdarah dengue',
        'icu': 'intensive care unit',
        'ugd': 'unit gawat darurat',
        'bpjs': 'badan penyelenggara jaminan sosial',
        'phbs': 'perilaku hidup bersih dan sehat',
        'psbb': 'pembatasan sosial berskala besar',

        # Pendidikan
        'sd': 'sekolah dasar',
        'smp': 'sekolah menengah pertama',
        'sma': 'sekolah menengah atas',
        'smk': 'sekolah menengah kejuruan',
        'ptn': 'perguruan tinggi negeri',
        'pts': 'perguruan tinggi swasta',
        'kkn': 'kuliah kerja nyata',
        'kpm': 'kuliah pengabdian masyarakat',
        'lppm': 'lembaga penelitian dan pengabdian masyarakat',
        'mbkm': 'merdeka belajar kampus merdeka',

        # Media & Jurnalistik
        'pwi': 'persatuan wartawan indonesia',
        'aji': 'aliansi jurnalis independen',
        'kpi': 'komisi penyiaran indonesia',
        'rri': 'radio republik indonesia',
        'tvri': 'televisi republik indonesia',
        'lkbn': 'lembaga kantor berita nasional',

        # Hukum & Peradilan
        'ma': 'mahkamah agung',
        'mk': 'mahkamah konstitusi',
        'kuhp': 'kitab undang undang hukum pidana',
        'kuhap': 'kitab undang undang hukum acara pidana',
        'uu': 'undang undang',
        'ham': 'hak asasi manusia',
        'lpsk': 'lembaga perlindungan saksi dan korban',

        # Lembaga & Program Regional
        'dprd': 'dewan perwakilan rakyat daerah',
        'kadin': 'kamar dagang dan industri',
        'gapensi': 'gabungan pelaksana konstruksi nasional indonesia',
        'hipmi': 'himpunan pengusaha muda indonesia',
        'apindo': 'asosiasi pengusaha indonesia',
        'koni': 'komite olahraga nasional indonesia',

        # Transportasi & Infrastruktur
        'damri': 'djawatan angkoetan motor republik indonesia',
        'pjka': 'perusahaan jawatan kereta api',
        'asdp': 'angkutan sungai danau dan penyeberangan',
        'pelni': 'pelayaran nasional indonesia',
        'hutama': 'husni tanra mighty associates',
        'wika': 'wijaya karya',
        'adhi': 'adhi karya',
        'pp': 'pembangunan perumahan',

        # Pertanian & Pangan
        'bulog': 'badan urusan logistik',
        'pusri': 'pupuk sriwidjaja',
        'pertani': 'perusahaan pertanian negara',
        'sang hyang seri': 'perusahaan umum benih nasional',
        'kostratani': 'komando strategis pembangunan pertanian',

        # Wilayah & Administrasi
        'dki': 'daerah khusus ibukota',
        'diy': 'daerah istimewa yogyakarta',
        'jabar': 'jawa barat',
        'jateng': 'jawa tengah',
        'jatim': 'jawa timur',
        'sumut': 'sumatera utara',
        'sumsel': 'sumatera selatan',
        'sulsel': 'sulawesi selatan',
        'sulteng': 'sulawesi tengah',
        'kalsel': 'kalimantan selatan',
        'kaltim': 'kalimantan timur',
        'ntb': 'nusa tenggara barat',
        'ntt': 'nusa tenggara timur',
        '苔蘭酱肉':'daging babi dengan saus tailan',
        '葱油汁蒸无骨黄金鲫':'ikan mas crucian emas tanpa tulang kukus dengan saus minyak daun bawang',
        '金奖灯笼百花稍梅' : 'lentera medali emas dengan ratusan bunga dan bunga prem',
        '金汤桃胶煨花胶' : 'ikan rebus permen karet persik sup emas maw',
        '青龙偃月刀' : 'pedang qinglong yanyue',
        '黄土高坡上的鱼子酱' : 'kaviar di lereng loess yang tinggi'
    }

def preprocess_text(text):
    text = re.sub(r'^[A-Za-z]*\s\(ANTARA\)\s-\s', '', text)
    text = str(text).lower()

    abbreviations = get_abbreviations_dict()
    words = text.split()
    expanded_words = []
    for word in words:
        if word.lower() in abbreviations:
            expanded_words.append(abbreviations[word.lower()])
        else:
            expanded_words.append(word)

    text = ' '.join(expanded_words)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    text = stopword_remover.remove(text)
    return text.strip()

def search_articles(query, df, vectorizer):
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    title_vectors = vectorizer.transform(df['judul_processed'].apply(str))
    
    similarities = cosine_similarity(query_vector, title_vectors).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                'judul': df['judul'].iloc[idx],
                'konten': df['konten'].iloc[idx],
                'similarity': similarities[idx],
                'kategori_utama': df['kategori_utama'].iloc[idx],
                'link': df['link'].iloc[idx]
            })
    return results

def main():
    st.title('Search Engine & Category Prediction')
    
    @st.cache_data
    def load_data():
        df = pd.read_csv('real-data-scraping.csv')
        df['judul_processed'] = df['judul'].apply(preprocess_text)
        df['konten_processed'] = df['konten'].apply(preprocess_text)
        return df
    
    @st.cache_resource
    def load_models():
        with open('vectorizer.pkl', 'rb') as f_vect:
            loaded_vectorizer = pickle.load(f_vect)

        with open('nb_model.pkl', 'rb') as f_model:
            loaded_model = pickle.load(f_model)
        return loaded_vectorizer, loaded_model
    
    try:
        df = load_data()
        vectorizer, nb_model = load_models()
        
        search_query = st.text_input('Enter your search query:')
        
        if search_query:
            results = search_articles(search_query, df, vectorizer)
            
            if results:
                st.subheader('Search Results:')
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result['judul']} (Similarity: {result['similarity']:.2f})"):
                        st.write("**Content:**")
                        st.write(result['konten'])

                        st.write("**Article Link:**")
                        st.markdown(f"[Read the full article]({result['link']})")
                        
                        processed_content = preprocess_text(result['konten'])
                        content_vector = vectorizer.transform([processed_content])
                        
                        predicted_category = nb_model.predict(content_vector)[0]
                        category_probs = nb_model.predict_proba(content_vector)[0]
                        categories = nb_model.classes_
                        
                        st.write('---')
                        st.write(f"**Actual Category:** {result['kategori_utama']}")
                        st.write(f"**Predicted Category:** {predicted_category}")
                        
                        st.write("**Top 3 Category Probabilities:**")
                        probs_df = pd.DataFrame({
                            'Category': categories,
                            'Probability': category_probs
                        }).sort_values('Probability', ascending=False).head(3)
                        
                        st.dataframe(probs_df)
            else:
                st.warning('No results found for your query.')
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure all required files (real-data-scraping.csv, vectorizer.pkl, nb_model.pkl) are available.")

if __name__ == '__main__':
    main()