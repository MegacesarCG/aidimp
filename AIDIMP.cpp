// Proyecto 1. Hecho por César Augusto Zabala De Freitas Barboza
// Red neuronal de Perceptron Simple que reconoce vocales
// Bautizado AIDIMP por "Artificial Inteligence Design Is My Passion"
// (El Diseño de Inteligencia Artificial Es Mi Pasion)

/* La entrada es un archivo de texto
Cada elemento esta separado por un espacio
Al final, opcionalmente, despues de una linea se coloca el resultado esperado donde:
1 0 0 0 0 = Es una A
0 1 0 0 0 = Es una E
0 0 1 0 0 = Es una I
0 0 0 1 0 = Es una O
0 0 0 0 1 = Es una U

Ejemplo: Para reconocer una A + resultado esperado

0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 1 1 1 1 0 0 0
0 0 1 1 1 1 1 1 0 0
0 1 1 1 0 0 1 1 1 0
0 1 1 0 0 0 0 1 1 0
0 1 1 0 0 0 0 1 1 0
0 1 1 0 0 0 0 1 1 0
0 1 1 0 0 0 0 1 1 0
0 1 1 0 0 0 0 1 1 0
0 1 1 0 0 0 0 1 1 0
0 1 1 1 0 0 1 1 1 0
0 0 1 1 1 1 1 1 1 0
0 0 0 1 1 1 1 0 1 1
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0

1 0 0 0 0

*/

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <time.h>
#include <fstream>

using namespace std;

static const int LENGTH = 16;
static const int WIDTH = 10;
static const int NO_NEURON = 5;
static const int TEST_ITERACIONES = 500;

static const float LEARNING_RATE = 0.3;
static const float UMBRALES[NO_NEURON] = {350,350,350,350,350};

static const string NOMBRE_QUERY_TXT = "letra.txt";
static const string NOMBRE_BASE_CONOCIMIENTO = "aidimpKnowledge.txt";
static const bool LEARNING = true;

bool first_time = true;

int r[NO_NEURON] = {1,0,0,0,0};
int a[LENGTH][WIDTH] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
						{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
						{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
						{0, 0, 1, 1, 1, 1, 1, 1, 1, 0},
						{0, 0, 0, 1, 1, 1, 1, 0, 1, 1},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						};
						
int r2[NO_NEURON] = {0,1,0,0,0};
int e[LENGTH][WIDTH] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
						{0, 1, 1, 1, 1, 1, 1, 1, 1, 0},
						{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 1, 1, 1, 0},
						{0, 1, 1, 1, 1, 1, 1, 1, 1, 0},
						{0, 1, 1, 1, 1, 1, 1, 1, 0, 0},
						{0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
						{0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
						{0, 1, 1, 1, 0, 0, 0, 0, 0, 0},
						{0, 0, 1, 1, 1, 1, 1, 1, 1, 0},
						{0, 0, 0, 1, 1, 1, 1, 1, 1, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						};

int r3[NO_NEURON] = {0,0,1,0,0};
int i[LENGTH][WIDTH] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						};
						
int r4[NO_NEURON] = {0,0,0,1,0};
int o[LENGTH][WIDTH] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
						{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
						{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
						{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
						{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						};

int r5[NO_NEURON] = {0,0,0,0,1};
int u[LENGTH][WIDTH] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 1, 1, 0, 0, 0, 1, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 0, 0, 0, 0, 1, 1, 0},
						{0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
						{0, 0, 1, 1, 1, 1, 1, 1, 0, 0},
						{0, 0, 0, 1, 1, 1, 1, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
						};

class Muestra{
	public:
		int entrada[LENGTH][WIDTH];
		int resultado[NO_NEURON];
		
		Muestra(){
			for(int i = 0; i < NO_NEURON; i++){
				resultado[i] = 0;
			}
		}
		
		void setEntrada(int x[LENGTH][WIDTH]){
			for(int i = 0; i < LENGTH; i++){
				for(int j = 0; j < WIDTH; j++){
					entrada[i][j] = x[i][j];
				}
			}
		}
		
		void setResultado(int r[NO_NEURON]){
			for(int i = 0; i < NO_NEURON; i++){
				resultado[i] = r[i];
			}
		}
		
		void imprimir(){
			cout << "Entrada = [" << endl;
			for(int i = 0; i < LENGTH; i++){
				cout << "[";
				for(int j = 0; j < WIDTH; j++){
					cout << entrada[i][j] << " ";
				}
				cout << "]" << endl;
			}
			cout << "]" << endl;
			
			cout << "Resultado esperado = [";
			for(int i = 0; i < NO_NEURON; i++){
				cout << resultado[i] << ", ";
			}
			cout << "]" << endl;
		}
};

Muestra muestras[NO_NEURON]; //Puesto aqui debido a que primero tenia que declarar la clase.

class Neurona{
	public:
		static const int RANDOM_LIMIT = 39;
		float umbral;
		
		float w[LENGTH][WIDTH];
		
		int inicializar(){
			for(int i = 0; i < LENGTH; i++){
				for(int j = 0; j < WIDTH; j++){
					w[i][j] = rand()%RANDOM_LIMIT;
				}
			}
			umbral = 0;
		};
		
		int inicializar(int x[LENGTH][WIDTH]);
		
		int procesar(int x[LENGTH][WIDTH]){
			int suma = 0;
			for(int i = 0; i < LENGTH; i++){
				for(int j = 0; j < WIDTH; j++){
					suma = suma + (w[i][j] * x[i][j]);
				}
			}
			return activacion(suma);
		}
		
		int activacion(int s){
			if(s - umbral > 0){
				return 1;
			}else{
				return 0;
			}
		}
		
		void imprimir(){
			cout << "Pesos = [" << endl;
			for(int i = 0; i < LENGTH; i++){
				cout << "[";
				for(int j = 0; j < WIDTH; j++){
					cout << w[i][j] << " ";
				}
				cout << "]" << endl;
			}
			cout << "]" << endl;
		}
		
		void actualizarPesos(int correcion, int x[LENGTH][WIDTH]){
			for(int i = 0; i < LENGTH; i++){
				for(int j = 0; j < WIDTH; j++){
					w[i][j] = w[i][j] + LEARNING_RATE * (correcion * x[i][j]);
				}
			}
		}
};

class Red{
	public:
		Neurona n[NO_NEURON];
		
		void inicializar(){
			ifstream database(NOMBRE_BASE_CONOCIMIENTO.c_str());
			//Existe la base de conocimiento
			if(database.good() == true){
				string line;
				int nlinea = 0; //# de linea actual
				int index = 0; //# de neurona actual
				string wi[WIDTH];
				while(getline(database,line)){
					//Indice de linea, usado para indicar la fila y cuando moverse a la siguiente neurona
					int i = nlinea%(LENGTH + 1); 
					if(i != LENGTH){
						separar(line, wi);
						//Asignar los pesos de la linea a la neurona
						for(int j = 0; j < WIDTH; j++){
							float weight = 0;
							sscanf(wi[j].c_str(), "%f", &weight);
							n[index].w[i][j] = weight;
						}
					}else{
						index = index + 1; //A la siguiente neurona
					}
					nlinea = nlinea + 1;
				}
				first_time = false;
			}else{
				for(int i = 0; i < NO_NEURON; i++){
					n[i].inicializar();
				}
			}
			for(int i = 0; i < NO_NEURON; i++){
					n[i].umbral = UMBRALES[i];
			}
		}
		
		void *separar(string linea, string separado[WIDTH]){
			size_t start;
    		size_t end = 0;
    		int i = 0;
 
    		while ((start = linea.find_first_not_of(" ", end)) != std::string::npos){
        		end = linea.find(" ", start);
        		separado[i] = linea.substr(start, end - start);
        		i = i + 1;
    		}
		}
		
		int *calcularError(int actual[NO_NEURON], int esperado[NO_NEURON]){
			for(int i = 0; i < NO_NEURON; i++){
				actual[i] = esperado[i] - actual[i];
			}
			return actual;
		}
		
		//Para identicar el valor uso una conversion binario decimal en donde cada resultado de una neurona representa un bit
		//Y ya que solo se pueden obtener un resultado, un solo bit debe estar activado.
		//NOTA: El numero binario es AL REVES, es decir [0,0,0,0,1] = [1,0,0,0,0] en la suma
		string identificar(int valor[NO_NEURON]){
			int suma = 0;
			for(int i = 0; i < NO_NEURON; i++){
				suma = suma + valor[i] * pow(2,i);
			}
			switch(suma){
				case 1: return "Esto es una A";
				break;
				case 2: return "Esto es una E";
				break;
				case 4: return "Esto es una I";
				break;
				case 8: return "Esto es una O";
				break;
				case 16: return "Esto es una U";
			}
			return "No se que letra es";
		}
		
		void primeraEjecucion(){
			
			for(int i = 0; i < TEST_ITERACIONES; i++){
				for(int casos = 0; casos < NO_NEURON; casos++){
					int resultadoActual[NO_NEURON];
					for(int j = 0; j < NO_NEURON; j++){
						resultadoActual[j] = n[j].procesar(muestras[casos].entrada);
					}
					int *correcion = calcularError(resultadoActual, muestras[casos].resultado);
					for(int j = 0; j < NO_NEURON; j++){
						n[j].actualizarPesos(correcion[j], muestras[casos].entrada);
					}
				}
			}
			cout << "*** Prueba final" << endl;
			for(int casos = 0; casos < NO_NEURON; casos++){
				cout << ">>> Caso #" << casos << endl;	
				int resultadoActual[NO_NEURON];
					for(int j = 0; j < NO_NEURON; j++){
						resultadoActual[j] = n[j].procesar(muestras[casos].entrada);
					}
					cout << "AIDIMP dice: \"" << identificar(resultadoActual) << "\"" << endl;
					cout << "Salida obtenida = [";
					for(int j = 0; j < NO_NEURON; j++){
						cout << resultadoActual[j]<< " ";
					}
					cout << "]" << endl;
					cout << "Salida esperada = [";
					for(int j = 0; j < NO_NEURON; j++){
						cout << muestras[casos].resultado[j] << " ";
					}
					cout << "]" << endl;
			}
		}
		
		void ejecutar(){
			Muestra query;
			ifstream qtext(NOMBRE_QUERY_TXT.c_str());
			bool test = false; // Si posee un resultado esperado entonces es una prueba
			if(qtext.good() == true){
				string line;
				int nlinea = 0; //# de linea actual
				string wi[WIDTH];
				while(getline(qtext,line)){
					int i = nlinea%(LENGTH + 1);
					if(test == false){
						if(i != LENGTH){
							separar(line, wi);
							//Asignar los pesos de la linea a la entrada
							for(int j = 0; j < WIDTH; j++){
								int value = 0;
								sscanf(wi[j].c_str(), "%d", &value);
								query.entrada[i][j] = value;
							}
						}else{
							test = true;
						}
					}else{
						if(!line.empty()){
							string res[NO_NEURON];
							separar(line, res);
							for(int j = 0; j < NO_NEURON; j++){
								int value = 0;
								sscanf(res[j].c_str(), "%d", &value);
								query.resultado[j] = value;
							}
						}else{
							test = false;
						}
					}
					nlinea = nlinea + 1;
				}
				//Ejecucion del programa
				int resultadoActual[NO_NEURON];
				for(int j = 0; j < NO_NEURON; j++){
					resultadoActual[j] = n[j].procesar(query.entrada);
				}
				cout << "AIDIMP dice: \"" << identificar(resultadoActual) << "\"" << endl;
				if(test == true){
					cout << "Salida obtenida = [";
					for(int j = 0; j < NO_NEURON; j++){
						cout << resultadoActual[j]<< " ";
					}
					cout << "]" << endl;
					cout << "Salida esperada = [";
					for(int j = 0; j < NO_NEURON; j++){
						cout << query.resultado[j] << " ";
					}
					cout << "]" << endl;
				}
				if(LEARNING == true && test == true){
					int *correcion = calcularError(resultadoActual, query.resultado);
					for(int j = 0; j < NO_NEURON; j++){
						n[j].actualizarPesos(correcion[j], query.entrada);
					}
				}
			}else{
				cout << "***ERROR***" << endl;
				cout << "No se ha detectado la query a procesar" << endl;
				cout << "Verifique que el archivo \"" << NOMBRE_QUERY_TXT << "\" se encuentre en la misma direccion que el programa" << endl;
			}
		}
		
		void guardar(){
			ofstream database;
			database.open(NOMBRE_BASE_CONOCIMIENTO.c_str(), fstream::out);
			for(int neurona = 0; neurona < NO_NEURON; neurona++){
				for(int i = 0; i < LENGTH; i++){
					for(int j = 0; j < WIDTH; j++){
						database << n[neurona].w[i][j] << " ";
					}
					database << endl;
				}
				database << endl;
			}
			database.close();
			cout << "Conocimiento guardado :)";
		}
};

int main()
{
	cout << "*** Bienvenido a AIDIMP ***" << endl;
	cout << "*** Artificial Inteligence Design Is My Passion ***" << endl;
	srand(time(NULL));
	Red red;
	red.inicializar();
	if(first_time == true){
		muestras[0].setEntrada(a);
		muestras[0].setResultado(r);
		muestras[1].setEntrada(e);
		muestras[1].setResultado(r2);
		muestras[2].setEntrada(i);
		muestras[2].setResultado(r3);
		muestras[3].setEntrada(o);
		muestras[3].setResultado(r4);
		muestras[4].setEntrada(u);
		muestras[4].setResultado(r5);
		red.primeraEjecucion();
	}else{
		red.ejecutar();
	}
	if(LEARNING == true){
		red.guardar();	
	}
    return 0;
}
