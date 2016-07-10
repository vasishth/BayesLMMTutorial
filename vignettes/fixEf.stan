data {
  int<lower=1> N;                //number of data points
  real rt[N];                    //reading time
  real<lower=-1,upper=1> so[N];  //predictor
}

parameters {
  vector[2] beta;            //intercept and slope
  real<lower=0> sigma_e;     //error sd
}

model {
  real mu;
  for (i in 1:N){                   // likelihood
    mu = beta[1] + beta[2]*so[i];
    rt[i] ~ lognormal(mu,sigma_e);
  }
}

