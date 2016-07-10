data {
	int<lower=1> N;
	real rt[N];                     //outcome
	real<lower=-1,upper=1> so[N];  //predictor
	int<lower=1> J;                  //number of subjects
	int<lower=1> K;                  //number of items
	int<lower=1, upper=J> subj[N];   //subject id
	int<lower=1, upper=K> item[N];   //item id
}

parameters {
	vector[2] beta;			// intercept and slopes
	real<lower=0> sigma_e;		// residual sd
	vector<lower=0>[2] sigma_u;	// subj sd
	vector<lower=0>[2] sigma_w;	// item sd
	cholesky_factor_corr[2] L_u;
	cholesky_factor_corr[2] L_w;
	matrix[2,J] z_u;
	matrix[2,K] z_w;
}

transformed parameters{
     	matrix[J,2] u;
	matrix[K,2] w;
     
     u = (diag_pre_multiply(sigma_u,L_u) * z_u)';	// subj random effects
	w = (diag_pre_multiply(sigma_w,L_w) * z_w)';	// item random effects
}

model {
	real mu;
	
	# priors:
	L_u ~ lkj_corr_cholesky(2.0);
	L_w ~ lkj_corr_cholesky(2.0);
	to_vector(z_u) ~ normal(0,1);
	to_vector(z_w) ~ normal(0,1);
	
	for (i in 1:N){
		mu = beta[1] + u[subj[i],1] + w[item[i],1] 
			+ (beta[2] + u[subj[i],2] + w[item[i],2])*so[i];
           rt[i] ~ lognormal(mu,sigma_e);        // likelihood
      }
}

generated quantities{
  real rt_tilde[N];
  real mu;
  for (i in 1:N){
    mu = beta[1] + u[subj[i],1] + w[item[i],1] 
        + (beta[2] + u[subj[i],2] + w[item[i],2])*so[i];
    rt_tilde[i] = lognormal_rng(mu,sigma_e);
  }
}
