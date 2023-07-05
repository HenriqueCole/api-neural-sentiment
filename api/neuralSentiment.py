import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Fazendo download do tokenizador (utilizado para dividar as palavras do texto)
nltk.download("punkt")
# Fazendo download do StopWords (palavras desnecessárias para análise, ex: a, de, um)
nltk.download("stopwords")


def preprocess_text(sentence):
    # Remove as pontuações do texto
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    # Troca todas as letras para minúscula
    sentence = sentence.lower()
    # Cria um vetor com as palavras do texto
    words = word_tokenize(sentence)
    # Filtra os StopWords do idioma português - PT
    stop_words = set(stopwords.words("portuguese"))
    # Remove do vetor de palavras todos os StopWords em PT
    words = [word for word in words if word not in stop_words]
    # Instanciando um objeto utilizado para transformação das palavras
    stemmer = PorterStemmer()
    # Transforma as palavras para o radical
    words = [stemmer.stem(word) for word in words]
    # Junta as palavras do vetor em uma única string
    sentence = " ".join(words)
    return sentence


train_data = [
    {"Eu amo este produto": "positivo"},
    {"Este produto é horrível": "negativo"},
    {"O filme foi incrível": "positivo"},
    {"Não gostei do serviço": "negativo"},
    {"A praia estava deslumbrante": "positivo"},
    {"A qualidade do produto é excelente": "positivo"},
    {"O serviço foi decepcionante": "negativo"},
    {"O desempenho do ator foi impressionante": "positivo"},
    {"A entrega foi atrasada e o produto veio quebrado": "negativo"},
    {"O show foi emocionante e cheio de energia": "positivo"},
    {"A equipe de suporte foi extremamente útil": "positivo"},
    {"A embalagem estava danificada e faltavam itens": "negativo"},
    {"A vista do topo da montanha era deslumbrante": "positivo"},
    {"O aplicativo é lento e cheio de bugs": "negativo"},
    {"O livro é envolvente e cativante": "positivo"},
    {"O serviço ao cliente é excelente": "positivo"},
    {"A comida no restaurante é deliciosa": "positivo"},
    {"O filme é tedioso e previsível": "negativo"},
    {"O hotel possui quartos confortáveis ​​e limpos": "positivo"},
    {"O transporte público é ineficiente e sempre atrasado": "negativo"},
    {"A apresentação do artista foi deslumbrante": "positivo"},
    {"O software é fácil de usar e possui recursos avançados": "positivo"},
    {"O concerto foi cancelado sem aviso prévio": "negativo"},
    {"O produto veio com defeito de fabricação": "negativo"},
    {"A equipe de vendas foi muito atenciosa": "positivo"},
    {"O processo de compra foi confuso e demorado": "negativo"},
    {"A exposição de arte foi inspiradora": "positivo"},
    {"O aplicativo travou várias vezes durante o uso": "negativo"},
    {"O produto superou todas as minhas expectativas": "positivo"},
    {"O serviço de entrega é rápido e eficiente": "positivo"},
    {"O restaurante tem um ambiente acolhedor e agradável": "positivo"},
    {"O desempenho do jogador foi decepcionante": "negativo"},
    {"O curso online oferece conteúdo de alta qualidade": "positivo"},
    {"O carro apresentou problemas mecânicos logo após a compra": "negativo"},
    {"O espetáculo teatral foi emocionante e envolvente": "positivo"},
    {"O atendimento ao cliente deixou a desejar": "negativo"},
    {"A trilha sonora do filme é cativante": "positivo"},
    {"O produto chegou antes do prazo estipulado": "positivo"},
    {"O professor explicou o conteúdo de forma clara e objetiva": "positivo"},
    {"O serviço de internet é instável e frequentemente cai": "negativo"},
    {"O parque de diversões tem atrações emocionantes para todas as idades": "positivo"},
    {"O hotel não cumpriu com as condições prometidas": "negativo"},
    {"A série de televisão é viciante e cheia de reviravoltas": "positivo"},
    {"O sistema de pagamento online apresentou falhas de segurança": "negativo"},
    {"O espetáculo de dança foi magnífico e cheio de graciosidade": "positivo"},
    {"O novo design do site é confuso e pouco intuitivo": "negativo"},
    {"O serviço de entrega não chegou dentro do prazo": "negativo"},
    {"O museu possui uma vasta coleção de obras de arte": "positivo"},
    {"O suporte técnico foi ineficiente na resolução do problema": "negativo"},
    {"O livro é uma obra-prima da literatura contemporânea": "positivo"},
    {"A música é contagiante e animada": "positivo"},
    {"A decoração do restaurante é elegante e sofisticada": "positivo"},
    {"O atendimento ao cliente foi rude e desrespeitoso": "negativo"},
    {"O filme é intrigante e cheio de suspense": "positivo"},
    {"O serviço de entrega cometeu erros graves e entregou o pedido errado": "negativo"},
    {"O parque tem atrações emocionantes para os aventureiros": "positivo"},
    {"O professor é dedicado e sempre disposto a ajudar os alunos": "positivo"},
    {"A qualidade do produto não corresponde ao preço cobrado": "negativo"},
    {"O concerto foi cancelado devido a problemas técnicos": "negativo"},
    {"O software possui uma interface intuitiva e fácil de usar": "positivo"},
    {"A festa de aniversário foi um verdadeiro fracasso de organização": "negativo"},
    {"O cenário natural é de tirar o fôlego e oferece vistas deslumbrantes": "positivo"},
    {"O ator principal entregou uma performance decepcionante": "negativo"},
    {"O serviço de transporte público é eficiente e pontual": "positivo"},
    {"O café da manhã no hotel é variado e delicioso": "positivo"},
    {"O produto apresenta durabilidade abaixo do esperado": "negativo"},
    {"A peça de teatro é uma mistura de comédia e drama": "positivo"},
    {"A empresa tem uma política de devolução injusta e complicada": "negativo"},
    {"O museu oferece uma experiência cultural enriquecedora": "positivo"},
    {"O aplicativo tem recursos inovadores, mas é instável e apresenta falhas": "negativo"},
    {"O restaurante é aclamado pela gastronomia de alta qualidade": "positivo"},
    {"O serviço de atendimento telefônico é demorado e pouco eficiente": "negativo"},
    {"O produto é resistente e durável, perfeito para uso diário": "positivo"},
    {"A trama do filme é previsível e não desperta interesse": "negativo"},
    {"A decisão dele foi muito ruim": "negativo"}
]

# TF = Quantidade de vezes que uma palavra aparece no texto / Quantidade de palavras do texto
# IDF = log(Quantidade de documentos / Quantidade de documentos que a palavra aparece)
# TF-IDF = TF * IDF (Quanto maior o valor, mais importante é a palavra para o texto)

# Serve para transformar o texto em um vetor de números, para que o classificador possa entender
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
# Vetorizando as frases para um formato compreensível para o treinamento
train_features = vectorizer.fit_transform(
    [list(x.keys())[0] for x in train_data])
# Obtendo os labels (positivo ou negativo) para o treinamento
train_labels = [list(x.values())[0] for x in train_data]
# Cria um modelo de classificação para encontrar o melhor hi
classifier = svm.SVC(kernel="linear")
# Treinando SVC para aumentar a margem entre as classes
classifier.fit(train_features, train_labels)


def predict_sentiment(sentence):
    # Realiza o pré-processamento da frase
    sentence = preprocess_text(sentence)
    # Vetoriza a frase para um formato compreensível para o classificador
    features = vectorizer.transform([sentence])
    # Classifica o sentimento da frase (positivo ou negativo)
    sentiment = classifier.predict(features)[0]
    return sentiment


print(predict_sentiment("Achei o produto muito ruim"))
