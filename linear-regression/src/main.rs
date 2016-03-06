extern crate csv;
extern crate rand;

use rand::distributions::{IndependentSample, Range};


fn main() {
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    let real_betas = vec![0.2, 1.1, 0.02, -1.2, 6.2];
    let real_bias = 0.32;

    let between = Range::new(-4.0, 4.0);
    let noise = Range::new(-1.0, 1.0);
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let mut point: Vec<f64> = Vec::new();
        for _ in 0..5 {
            point.push(between.ind_sample(&mut rng))
        }
        let y_actual = point.iter()
        .zip(real_betas.iter())
        .fold(real_bias, |sum, (&b, &x)| sum + (x * b) + noise.ind_sample(&mut rng));

        x.push(point);
        y.push(y_actual);
    }

    let mut regression = LinearRegression::new(x, y);
    regression.fit(0.001, 1000);
    println!("Finished: Params: {}, {}, {}, {}, {}", regression.betas[0], regression.betas[1], regression.betas[2], regression.betas[3], regression.betas[4])

}



struct LinearRegression {
    betas: Vec<f64>,
    bias: f64,
    x: Vec<Vec<f64>>,
    y: Vec<f64>
}

impl LinearRegression {
    // b: number of parameters for regression
    fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> LinearRegression {
        let m = x[0].len();
        LinearRegression { betas: vec![0.0; m], bias: 1.0, x: x, y: y }
    }

    fn step(&mut self, i: usize, step_size: f64) {
        let y_hat = self.x[i].iter()
            .zip(self.betas.iter())
            .fold(self.bias, |sum, (&x, &b)| sum + (x * b));
        let err = y_hat - self.y[i];

        self.betas = self.betas.iter()
            .zip(self.x[i].iter())
            .map(|(&b, x)| {
                b - (step_size * 2.0 * x * err)
            })
            .collect();
        self.bias -= 2.0 * step_size * err;
    }

    fn fit(&mut self, step_size: f64, num_iterations: usize) {
        for _ in 0..num_iterations {
            for i in 0..self.x.len() {
                self.step(i, step_size);
            }
        }
    }

    fn predict(self, x: Vec<f64>) -> f64 {
        x.iter()
            .zip(self.betas.iter())
            .fold(self.bias, |sum, (x, b)| sum + (x * b))
    }
}
