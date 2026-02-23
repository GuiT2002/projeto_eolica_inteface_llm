from groq import Groq
import re
import wind_farm_GA_16 as simulador

client = Groq(api_key='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')

system_prompt = """

Você é um assistente que irá auxiliar o usuário a utilizar um simulador de posições de turbinas em parques eólicos. 
A sua tarefa é obter do usuário as informações necessárias para a simulação. 
Para realizar isso, cada uma de suas respostas deverá na seguinte estrutura definida por tags:

<tabela>
CUT_IN_WIND_SPEED:      -- a menor velocidade do vento (em m/s) na qual a turbina começa a gerar energia útil. Default é 9.8. DEVE SEMPRE ESTAR EM M/S
CUT_OUT_WIND_SPEED:     -- a velocidade do vento (em m/s) a partir da qual a turbina para/desliga para proteção. Default é 25. DEVE SEMPRE ESTAR EM M/S
RATED_WIND_SPEED:       -- Velocidade nominal do vento em m/s. Default é 9.8. DEVE SEMPRE ESTAR EM M/S
RATED_POWER:            -- Potência nominal da turbina em watts. Default é 3350000. DEVE SEMPRE ESTAR EM W
TURB_DIAM:              -- Diâmetro da turbina em metros. Default é 100. DEVE SEMPRE ESTAR EM METROS
CIRCLE_RADIUS:          -- indica o raio do círculo em metros que delimita a área de simulação. Precisa necessariamente ser informado pelo usuário. DEVE SEMPRE ESTAR EM METROS
IND_SIZE:               -- indica o número de turbinas que deseja-se incluir na simulação. Precisa necessariamente ser informado pelo usuário. DEVE SEMPRE SER APENAS UM ÚNICO NÚMERO INTEIRO
N_DIAMETERS:            -- número mínimo de diâmetros das turbinas (em metros) que irá separar cada uma na simulação. Default é 3. DEVE SEMPRE ESTAR EM METROS
<tabela_f>


<mensagem>
-- a sua mensagem para o usuário deve estar aqui. **Coloque a sua mensagem que vai aparecer para o usuário entre estas tags.
<mensagem_f>


<status>
-- código de status que será definido a seguir.
<status_f>

Sua tarefa é preencher os valores que estão entre as tags "tabela" e "tabela_f", adicionando ao lado deles o valor de cada campo que o usuário especificar.
Caso alguma informação obrigatória não esteja presente e o usuário decida iniciar a simulação, você deve informar antes que está faltando uma informação obrigatória.
A tabela que está entre <tabela> e <tabela_f> deve ser preenchida apenas com valores númericos, não adicione nenhuma unidade de medida ao lado e converta o valor que o usuário colocar para a unidade de medida especificada na descrição da tabela.
As suas respostas devem ser breves e sempre voltadas à simulação. Após perguntar sobre os campos obrigatórios, você deve dizer ao usuário quais são os outros campos que permitem alterações e explicar o que cada um deles significa, além de apresentar seus valores default. 

A tabela com as descrições dos códigos de status é a seguinte:
0 -- indica que o usuário deseja fechar o programa e o simulador.
1 -- indica que o usuário ainda está preenchendo as informações.
2 -- indica que todas as informações necessárias foram obtidas e o usuário decidiu iniciar a simulação.

Apenas um desses números deverá estar entre as tags "status" e "status_f".
Você só deve iniciar a simulação e mudar o código de status para 2 após o usuário pedir. Mude o código de status para 0 apenas se o usuário pedir.
Quando o usuário desejar iniciar a simulação, DIGA SOMENTE "iniciando a simulação" e mude o código de status para 2. Caso o usuário queira mudar algo após a simulação terminar, você deve mudar o código de status para 1.

"""

def _extract_between(response: str, start_tag: str, end_tag: str) -> str:
    m = re.search(rf"<{start_tag}>\s*(.*?)\s*<{end_tag}>", response, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def extract_table(response: str):
    """
    Retorna:
      - first5: tupla com os 5 primeiros valores; os 3 primeiros são float e os 2 seguintes são int
      - circle_radius: int
      - ind_size: int
      - n_diameters: int
    Ordem dos 3 últimos: CIRCLE_RADIUS, IND_SIZE, N_DIAMETERS
    """
    table_block = _extract_between(response, "tabela", "tabela_f")
    nums = re.findall(r"-?\d+(?:\.\d+)?", table_block)

    # Mantém o valor numérico; depois converte seletivamente
    vals = [float(n) for n in nums]

    first5_list = vals[:5]
    # 3 primeiros float, 4º e 5º int
    first5 = (float(first5_list[0]), float(first5_list[1]), float(first5_list[2]),
              int(first5_list[3]), int(first5_list[4])) if len(first5_list) == 5 else tuple(first5_list)

    circle_radius = int(vals[5]) if len(vals) > 5 else None
    ind_size = int(vals[6]) if len(vals) > 6 else None
    n_diameters = int(vals[7]) if len(vals) > 7 else None

    return first5, circle_radius, ind_size, n_diameters

def extract_message(response: str):
    """
    Retorna: string dentro de <mensagem>...</mensagem_f>.
    """
    return _extract_between(response, "mensagem", "mensagem_f")

def extract_status(response: str):
    """
    Retorna: inteiro dentro de <status>...</status_f>.
    """
    status_block = _extract_between(response, "status", "status_f")
    m = re.search(r"-?\d+", status_block)
    return int(m.group(0)) if m else None


messages = [
        {
            "role": "system",
            "content": f"{system_prompt}"
        },
        {
            "role": "user",
            "content": "Olá",
        }
    ]

status = 10

while status != 0:

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.4
    )

    messages.append({"role": "assistant", "content": chat_completion.choices[0].message.content})

    print(f"\033[92mAssistente: {extract_message(chat_completion.choices[0].message.content)}\033[0m")

    status = extract_status(chat_completion.choices[0].message.content)

    turb, circle_radius, ind_size, n_diameters = extract_table(chat_completion.choices[0].message.content)

    if status == 0:
        break

    if status == 2:
        print(f"""\033[92m
CUT_IN_WIND_SPEED:           {turb[0]} m/s   
CUT_OUT_WIND_SPEED:          {turb[1]} m/s   
RATED_WIND_SPEED:            {turb[2]} m/s     
RATED_POWER:                 {turb[3]} W         
TURB_DIAM:                   {turb[4]} m           
CIRCLE_RADIUS:               {circle_radius} m    
IND_SIZE:                    {ind_size} turbinas
N_DIAMETERS:                 {n_diameters} m\033[0m""")
        simulador.run_ga(TURB_ATRBT_DATA=turb, CIRCLE_RADIUS=circle_radius, IND_SIZE=ind_size, N_DIAMETERS=n_diameters)

    user_input = str(input("\nUsuário: "))

    messages.append({"role": "user", "content": user_input})


