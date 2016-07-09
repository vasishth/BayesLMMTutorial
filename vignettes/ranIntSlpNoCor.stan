data {
  int<lower=1> N;                  //number of data points
  real rt[N];                      //reading time
  real<lower=-1,upper=1> so[N];    //predictor
  int<lower=1> J;                  //number of subjects
  int<lower=1> K;                  //number of items
  int<lower=1, upper=J> subj[N];   //subject id
  int<lower=1, upper=K> item[N];   //item id
}

parameters {
  vector[2] beta;                  //intercept and slope
  real<lower=0> sigma_e;           //error sd
  matrix[2,J] u;                   //subj intercepts, slopes
  vector<lower=0>[2] sigma_u;      //subj sd
  matrix[2,K] w;                   //item intercepts, slopes
  vector<lower=0>[2] sigma_w;      //item sd
}

model {
  real mu;
  //priors
  for (i in 1:2){
    u[i] ~ normal(0,sigma_u[i]);    //subj random effects
    w[i] ~ normal(0,sigma_w[i]);    //item random effects
  }
  //likelihood
  for (i in 1:N){
    mu = beta[1] + u[1,subj[i]] + w[1,item[i]] 
          + (beta[2] + u[2,subj[i]] + w[2,item[i]])*so[i];
    rt[i] ~ lognormal(mu,sigma_e);
  }
}
