#define DEBUG 0

#include <Wire.h>
#include <Adafruit_MMA8451.h>
#include <arduinoFFT.h>

Adafruit_MMA8451 mma = Adafruit_MMA8451();


// FFT Configuration
const uint16_t TIME_SAMPLES = 800;
const double   SAMPLING_FREQ = 800;         // MMA running at highest (800Hz)
const uint16_t FREQ_SAMPLES = 512;        // These must be a power of 2
const uint16_t FREQ_BINS_N  = (int)(FREQ_SAMPLES / 2);


// Vectors 
float a[TIME_SAMPLES];  // Accelerations (Time Domain)
float f[FREQ_BINS_N];   // Frequencies
float m[FREQ_BINS_N];   // Magnitudes

// Used for Fourier Transform
float vReal[FREQ_SAMPLES];
float vImag[FREQ_SAMPLES];



// FFT object
ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, FREQ_SAMPLES, SAMPLING_FREQ);

void setup() {
  Serial.begin(115200);
  while (!Serial); 

  if (!mma.begin()) {
    Serial.println("Could not start MMA8451");
    while (1);
  }

  mma.setRange(MMA8451_RANGE_2_G);
  mma.setDataRate(MMA8451_DATARATE_800_HZ);
  Serial.println("MMA8451 Found. Starting Sampling...");
}

void loop() {
  // Collects samples over 1 second (800 at 800Hz), Computes Frequences & Magnitudes using FFT 
  // Stored in static vars a (acceleration), f (frequencies), m (magnitudes)
  data_acq();

  // Perform Feature Extractions:
  float MU = sum(a, TIME_SAMPLES) / TIME_SAMPLES;

  float rms = RMS();
  float std = STD(MU);
  float skew = skewness(MU, std);
  float kurt = kurtosis(MU, std);
  float ptp = PTP();
  float cf = crest(rms);

  float sc = spectral_centroid();
  float ss = spectral_spread(sc);
  float se = spectral_energy();
  float sen = spectral_entropy();
  float spf = spectral_peak(); 

  
  Serial.println();
  Serial.print("rms  : "); Serial.println(rms);
  Serial.print("std  : "); Serial.println(std);
  Serial.print("skew : "); Serial.println(skew);
  Serial.print("kurt : "); Serial.println(kurt);
  Serial.print("ptp  : "); Serial.println(ptp);
  Serial.print("cf   : "); Serial.println(cf);
  Serial.print("sc   : "); Serial.println(sc);
  Serial.print("ss   : "); Serial.println(ss);
  Serial.print("se   : "); Serial.println(se);
  Serial.print("sen  : "); Serial.println(sen);
  Serial.print("spf  : "); Serial.println(spf);

  Serial.println("---------------------------------");
  delay(10000); // Go to sleep or 10 mins
}


void data_acq() {
  float sum_mag = 0;

  // 1. DATA ACQUISITION
  // We need to sample as fast as the sensor allows
  for (int i = 0; i < TIME_SAMPLES; i++) {
    unsigned long microPeriod = micros();

    mma.read(); // GET NEW DATA
    
    // Convert to G's
    float x = (float)mma.x / 4096.0;
    float y = (float)mma.y / 4096.0;
    float z = (float)mma.z / 4096.0;

    // Compress trixial (Vector Sum) 
    a[i] = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
    if (i < FREQ_SAMPLES){
      // Freq & Magnitude
      vReal[i] = a[i];
      vImag[i] = 0.0;
      sum_mag += vReal[i];
    }

    // Wait for the next 800Hz sample period (1250 microseconds)
    while(micros() - microPeriod < (1000000 / SAMPLING_FREQ)); 
  }

  vReal[0] = 0;

  // 2. PRE-PROCESSING (Remove DC Offset / Gravity)
  // This centers the signal around 0.0 so 0Hz isn't huge.
  float mean = sum_mag / FREQ_SAMPLES;
  for (int i = 0; i < FREQ_SAMPLES; i++) {
    vReal[i] -= mean;
  }

  // 3. FFT PROCESSING
  // If leak (features like centroid begins drifitng even though vibration is consistent) then try FFT_WIN_TYP_BLACKMAN_HARRIS
  FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD); // Smoothes the edges of the data
  FFT.compute(FFT_FORWARD);                        // Calculate FFT
  FFT.complexToMagnitude();                        // Convert to usable magnitudes

  // 4. FIND PEAK FREQUENCY
  if (DEBUG) {
    float peak = FFT.majorPeak();
    Serial.print("Major Peak: ");
    Serial.print(peak);
    Serial.println(" Hz");
  }

  // Set Frequencies & Magnitude vectors

  for (int i = 0; i < (FREQ_BINS_N); i++) {
    // Freq Bin Formula: Index * SamplingFrequency / TotalFREQ_Samples
    float freq = (i * SAMPLING_FREQ) / FREQ_SAMPLES;
    // Magnitudes are stored in vReal from ArduinoFFT.complexToMagnitude(...)
    float mag = vReal[i];
    
    f[i] = freq;
    m[i] = mag;
  }

  if (DEBUG) {
    // 5. OUTPUT RESULTS
    Serial.println("Frequency (Hz) | Magnitude");
    for (int i = 0; i < (FREQ_BINS_N); i++) {
      // Formula: Index * SamplingFrequency / TotalFREQ_Samples
      float freq = f[i];
      float mag = m[i];

      // Only print frequencies that actually have a signal (noise floor cut-off)
      if (mag > 0.05) { 
        Serial.print("~ ");
      }
      Serial.print(freq, 1);
      Serial.print(" Hz: \t");
      Serial.println(mag, 4);
      
    }
  }
}



float sum(float* vec, unsigned int len){
  float sum = 0;
  for (int i = 0; i < len; i++)
    sum += vec[i];
  return sum;
}


/* Time Domain Features */
float RMS() {
  float a_squared[TIME_SAMPLES]; 
  for (int i = 0; i < 800; i++){
    a_squared[i] = pow(a[i], 2);
  };
  return sqrt(sum(a_squared, TIME_SAMPLES) / TIME_SAMPLES);
}

float STD(float MU) {
  float inner[TIME_SAMPLES];
  for (int i = 0; i < TIME_SAMPLES; i++){
    inner[i] = pow(a[i] - MU, 2);
  }
  return sqrt(sum(inner, TIME_SAMPLES) / TIME_SAMPLES);
}

float skewness(float MU, float STD) {
  float numerator_inner[TIME_SAMPLES];
  for (int i = 0; i < TIME_SAMPLES; i++){
    numerator_inner[i] = pow(a[i] - MU, 3);
  } 
  float numerator = sum(numerator_inner, TIME_SAMPLES);
  float denominator = (TIME_SAMPLES-1)*pow(STD,3);
  return numerator / denominator;
}

float kurtosis(float MU, float STD) {
  float numerator_inner[TIME_SAMPLES];
  for (int i = 0; i < TIME_SAMPLES; i++){
    numerator_inner[i] = pow(a[i] - MU, 4);
  } 
  float denominator = (TIME_SAMPLES-1)*pow(STD,4);
  float numerator = sum(numerator_inner, TIME_SAMPLES);
  return numerator / denominator;
}

float PTP(){
  float a_max = a[0];
  float a_min = a[0];

  for (int i = 0; i < TIME_SAMPLES; i++){
    if (a[i] > a_max) a_max = a[i];
    if (a[i] < a_min) a_min = a[i];
  }
  return a_max - a_min;
}

float crest(float RMS) {
  float a_abs_max = a[0];
  
  for (int i = 0; i < TIME_SAMPLES; i++){
    if (abs(a[i]) > a_abs_max) a_abs_max = abs(a[i]);
  }
  return a_abs_max / RMS;
}

/* Frequency Domain Features */
float spectral_centroid() {
  float numerator_sum = 0;
  for (int n = 0; n < FREQ_BINS_N; n++) {
    numerator_sum += f[n] * m[n];
  }
    
  return numerator_sum / sum(m, FREQ_BINS_N);
}

/* Takes Spectral Centroid (sc)*/
float spectral_spread(float SC) {
  float numerator_sum = 0;

  for (int n = 0; n < FREQ_BINS_N; n++) {
    numerator_sum += (pow((f[n] - SC), 2) * m[n]);
  }
  
  return sqrt(numerator_sum / sum(m, FREQ_BINS_N));
}

float spectral_energy() {
  float m_sqd_sum = 0;
  for (int n = 0; n < FREQ_BINS_N; n++) {
    m_sqd_sum += pow(m[n], 2);
  }
  return m_sqd_sum;
}

float spectral_entropy() {
  float m_sum = sum(m, FREQ_BINS_N);
  float p[FREQ_BINS_N];

 

  // Final step (Entire Computation)
  float numerator_sum = 0;
  for (int n = 0; n < FREQ_BINS_N; n++) {
    float probability_distribution_n = m[n] / m_sum;
    if (probability_distribution_n > 0)
      // Safe approach - Where P_n <= 0: P_n = avg(p)
      numerator_sum += probability_distribution_n * log2(probability_distribution_n);
  }
  numerator_sum = -numerator_sum;

  return numerator_sum / log2(FREQ_BINS_N);
}

float spectral_peak() {
  float m_max = m[0];
  unsigned int f_argmax_m = 0;
  for (int n = 0; n < FREQ_BINS_N; n++) {
    if (m[n] > m_max) {
      m_max = m[n];
      f_argmax_m = f[n];
    }
  }
  return f_argmax_m;
}

