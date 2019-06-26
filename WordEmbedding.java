import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

public class WordEmbedding {

	public static void main(String[] args) throws Exception {
		if (args.length == 0) {
			System.out.println("Uso: java WordEmbedding [SBW-vectors-300-min5.bin]");
			System.out
					.println("  Bajar desde http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz");
			System.out.println("  y descomprimir con gunzip");
			System.out.println("  Más detalles en https://crscardellino.github.io/SBWCE/");
			return;
		}
		File file = new File(args[0]);
		if (!file.exists()) {
			System.out.println("No puedo leer archivo " + args[0]);
			return;
		}
		WordEmbedding embedding = new WordEmbedding(file);
		// imprimir distancia entre pares
		embedding.printDistance("instituto", "universidad");
		embedding.printDistance("instituto", "casa");
		embedding.printDistance("perro", "gato");
		embedding.printDistance("perro", "mascota");
		embedding.printDistance("perro", "lobo");
		embedding.printDistance("perro", "camión");
		// imprimir los vectores mas cercanos
		embedding.printWordKnn("Chile", 5);
		embedding.printWordKnn("chile", 5);
		embedding.printWordKnn("CHILE", 5);
		embedding.printWordKnn("Santiago", 5);
		embedding.printWordKnn("santiago", 5);
		embedding.printWordKnn("numero", 5);
		embedding.printWordKnn("número", 5);
		embedding.printWordKnn("pagina", 5);
		embedding.printWordKnn("página", 5);
		embedding.printWordKnn("compro", 5);
		embedding.printWordKnn("compró", 5);
		embedding.printWordKnn("zapatos", 5);
		embedding.printWordKnn("rojo", 5);
		// descubrir una palabra por analogía
		embedding.testAnalogia("hombre", "mujer", "rey", "reina");
		embedding.testAnalogia("hermano", "hermana", "nieto", "nieta");
		embedding.testAnalogia("ingeniero", "ingeniera", "doctor", "doctora");
		embedding.testAnalogia("Francia", "París", "Rusia", "Moscú");
		embedding.testAnalogia("perro", "perros", "camión", "camiones");
		embedding.testAnalogia("feliz", "felizmente", "lento", "lentamente");
		embedding.testAnalogia("bonito", "feo", "rápido", "lento");
		embedding.printResultadoAnalogias();
	}

	private int numPalabras;
	private int dimensiones;
	private int contAnalogiasCorrectas;
	private int contAnalogiasCercanas;
	private int contAnalogiasIncorrectas;
	private int contAnalogiasDesconocidas;
	private ArrayList<String> allPalabras;
	private float[] allVectors;
	private Map<String, Integer> mapPalabraToId;

	public WordEmbedding(File file) throws Exception {
		long start = System.currentTimeMillis();
		InputStream is = new BufferedInputStream(new FileInputStream(file));
		System.out.println("leyendo " + file);
		this.numPalabras = Integer.parseInt(readNextString(is));
		this.dimensiones = Integer.parseInt(readNextString(is));
		System.out.println("palabras=" + numPalabras + " dim=" + dimensiones);
		this.allPalabras = new ArrayList<String>();
		this.allPalabras.ensureCapacity(numPalabras);
		// array con todos los vectores juntos
		this.allVectors = new float[numPalabras * dimensiones];
		this.mapPalabraToId = new HashMap<String, Integer>(numPalabras * 2);
		for (int i = 0; i < numPalabras; ++i) {
			String word = readNextString(is);
			readFloatArray(is, allVectors, i * dimensiones, dimensiones);
			this.allPalabras.add(word);
			this.mapPalabraToId.put(word, i);
		}
		long end = System.currentTimeMillis();
		is.close();
		System.out.println("cargado OK " + (end - start) + " ms");
	}

	private static final String readNextString(InputStream is) throws Exception {
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		for (;;) {
			int c = is.read();
			if (c == '\n' || c == ' ')
				break;
			bos.write(c);
		}
		return bos.toString(StandardCharsets.UTF_8.name());
	}

	private static final float readFloat(InputStream is) throws Exception {
		int b0 = is.read();
		int b1 = is.read();
		int b2 = is.read();
		int b3 = is.read();
		int bits = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
		return Float.intBitsToFloat(bits);
	}

	private static final void readFloatArray(InputStream is, float[] array, int idx_start, int largo_array)
			throws Exception {
		for (int j = 0; j < largo_array; ++j) {
			float f = readFloat(is);
			array[idx_start + j] = f;
		}
	}

	public void printDistance(String palabra1, String palabra2) {
		int id1 = buscarIdPalabra(palabra1);
		int id2 = buscarIdPalabra(palabra2);
		if (id1 < 0 || id2 < 0)
			return;
		double dist = distanciaEntrePalabras(id1, id2);
		System.out.println("dist(" + palabra1 + ", " + palabra2 + ") = " + dist);
	}

	public void testAnalogia(String palabra1a, String palabra1b, String palabra2a, String palabra2b) {
		System.out.println("\"" + palabra1a + "\" es a \"" + palabra1b + "\" como \"" + palabra2a + "\" es a:");
		int id1a = buscarIdPalabra(palabra1a);
		int id1b = buscarIdPalabra(palabra1b);
		int id2a = buscarIdPalabra(palabra2a);
		int id2b = buscarIdPalabra(palabra2b);
		if (id1a < 0 || id1b < 0 || id2a < 0 || id2b < 0) {
			contAnalogiasDesconocidas += 1;
			return;
		}
		// la analogia dice: (palabra_1a - palabra_1b) = (palabra_2a - palabra_2b)
		// despejando 2b queda: palabra_2b = (palabra_1b - palabra_1a + palabra_2a)
		float[] vector = new float[dimensiones];
		// palabra_1b
		sumarPalabra(vector, id1b, 1);
		// restar palabra_1a
		sumarPalabra(vector, id1a, -1);
		// sumar palabra_2a
		sumarPalabra(vector, id2a, 1);
		// buscar los mas cercanos ignorando las palabras conocidas
		int posicion = printWordKnn(vector, 3, id1a, id1b, id2a, id2b);
		// verificar si es la mas cercana
		if (posicion == 0)
			contAnalogiasCorrectas += 1;
		else if (posicion > 0)
			contAnalogiasCercanas += 1;
		else
			contAnalogiasIncorrectas += 1;
	}

	public void printResultadoAnalogias() {
		System.out.println("analogias correctas=" + contAnalogiasCorrectas + " incorrectas=" + contAnalogiasIncorrectas
				+ " cercanas=" + contAnalogiasCercanas + " desconocidas=" + contAnalogiasDesconocidas);
	}

	public void printWordKnn(String palabraQuery, int knn) {
		int idPalabraQuery = buscarIdPalabra(palabraQuery);
		if (idPalabraQuery < 0)
			return;
		float[] vector = new float[dimensiones];
		sumarPalabra(vector, idPalabraQuery, 1);
		System.out.println(palabraQuery);
		// buscar los vecinos mas cercanos
		printWordKnn(vector, knn, idPalabraQuery, -1, -1, -1);
	}

	private static final class PalabraCercana {
		final double distancia;
		final int idPalabra;

		public PalabraCercana(double distancia, int idPalabra) {
			this.distancia = distancia;
			this.idPalabra = idPalabra;
		}

	}

	public int printWordKnn(float[] queryVector, int knn, int idIgnorar1, int idIgnorar2, int idIgnorar3,
			int idCorrecta) {
		// cola de prioridad de mayor a menor distancia
		PriorityQueue<PalabraCercana> queue = new PriorityQueue<PalabraCercana>(new Comparator<PalabraCercana>() {
			@Override
			public int compare(PalabraCercana o1, PalabraCercana o2) {
				return Double.compare(o2.distancia, o1.distancia);
			}
		});
		// calcular la distancia mas cercana
		for (int id = 0; id < numPalabras; ++id) {
			if (id == idIgnorar1 || id == idIgnorar2 || id == idIgnorar3)
				continue;
			double dist = distanciaEntreVectorYPalabra(queryVector, id);
			if (queue.size() < knn) {
				// agregar a la cola los k primeros
				queue.add(new PalabraCercana(dist, id));
			} else if (dist < queue.peek().distancia) {
				// si tiene una distancia mejor que el peor candidato
				// sacar el peor candidato
				queue.remove();
				// agregar esta palabra
				queue.add(new PalabraCercana(dist, id));
			}
		}
		// en la cola de prioridad estan en orden invertido (peor a mejor)
		StringBuilder sb = new StringBuilder();
		// invertir el orden
		int posicion = -1;
		while (queue.size() > 0) {
			PalabraCercana entry = queue.remove();
			String linea = allPalabras.get(entry.idPalabra) + " " + entry.distancia;
			sb.insert(0, "  " + (idCorrecta == entry.idPalabra ? "->" : "") + linea + "\n");
			if (idCorrecta == entry.idPalabra)
				posicion = queue.size();
		}
		// imprimir los resultados ordenados
		System.out.print(sb.toString());
		return posicion;
	}

	private void sumarPalabra(float[] vectorPalabra, int idPalabra, float multiplicador) {
		int idx_start = idPalabra * dimensiones;
		for (int i = 0; i < dimensiones; ++i) {
			vectorPalabra[i] += allVectors[idx_start + i] * multiplicador;
		}
	}

	private int buscarIdPalabra(String palabra) {
		Integer id = mapPalabraToId.get(palabra);
		if (id == null) {
			System.out.println("no encuentro palabra \"" + palabra + "\"");
			return -1;
		}
		return id.intValue();
	}

	private double distanciaEntrePalabras(int idPalabra1, int idPalabra2) {
		float[] vectorPalabra = new float[dimensiones];
		// obtener el vector de la palabra 1
		sumarPalabra(vectorPalabra, idPalabra1, 1);
		return distanciaEntreVectorYPalabra(vectorPalabra, idPalabra2);
	}

	private double distanciaEntreVectorYPalabra(float[] vectorPalabra, int idPalabra) {
		float sum = 0;
		float norma1 = 0;
		float norma2 = 0;
		int idx_start = idPalabra * dimensiones;
		// distancia coseno
		for (int i = 0; i < dimensiones; i++) {
			// multiplicacion de pesos
			sum += vectorPalabra[i] * allVectors[idx_start + i];
			// norma vector v
			norma1 += vectorPalabra[i] * vectorPalabra[i];
			// norma vector id_word
			norma2 += allVectors[idx_start + i] * allVectors[idx_start + i];
		}
		// similitud
		double cos_sim = sum / (Math.sqrt(norma1) * Math.sqrt(norma2));
		// convertir a distancia. Se podria hacer sqrt(2*(1-cos))
		return 1 - cos_sim;
		// distancia euclideana
		// for (int i = 0; i < vectorDimensions; i++) {
		// float f = Math.abs(v[i] - allVectors[idx_start + i]);
		// sum += f * f;
		// }
		// return Math.sqrt(sum);
	}

}
