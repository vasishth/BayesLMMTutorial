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
  vector<lower=0>[2] sigma_u;      //subj sd
  vector<lower=0>[2] sigma_w;      //item sd
  cholesky_factor_corr[2] L_u;
  cholesky_factor_corr[2] L_w;
  matrix[2,J] z_u;
  matrix[2,K] z_w;
}

transformed parameters{
  matrix[2,J] u;
  matrix[2,K] w;
  
  u = diag_pre_multiply(sigma_u,L_u) * z_u;	//subj random effects
  w = diag_pre_multiply(sigma_w,L_w) * z_w;	//item random effects
}

model {
  real mu;
  //priors
  L_u ~ lkj_corr_cholesky(2.0);
  L_w ~ lkj_corr_cholesky(2.0);
  to_vector(z_u) ~ normal(0,1);
  to_vector(z_w) ~ normal(0,1);
  //likelihood
  for (i in 1:N){
    mu = beta[1] + u[1,subj[i]] + w[1,item[i]] 
          + (beta[2] + u[2,subj[i]] + w[2,item[i]])*so[i];
    rt[i] ~ lognormal(mu,sigma_e);
  }
}
