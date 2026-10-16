#!/usr/bin/env Rscript
# Fit a separate participant-level binomial GLMM for each supplied model name.
args <- commandArgs(trailingOnly=TRUE)
sensitivity <- '--sensitivity' %in% args
args <- args[args != '--sensitivity']
if (length(args) < 2) stop('Usage: Rscript fit_alignment_moderation.R OUTPUT_DIR MODEL [MODEL...]')
extra_lib <- Sys.getenv('FOCUS_R_LIBRARY')
if (nzchar(extra_lib)) .libPaths(c(extra_lib, .libPaths()))
suppressPackageStartupMessages(library(lme4))
library(jsonlite)
out <- normalizePath(args[1], mustWork=TRUE)
dat <- read.csv(file.path(out,'analysis_data.csv'), stringsAsFactors=FALSE)
full_formula <- neg ~ prediction_z * alignment_c + (1+prediction_z||participant) + (1+prediction_z||context) + (1|item)
null_formula <- neg ~ prediction_z + alignment_c + (1+prediction_z||participant) + (1+prediction_z||context) + (1|item)

fit_once <- function(formula, d, optimizer, strict=FALSE, start=NULL) {
  warnings <- character()
  fit <- withCallingHandlers(glmer(formula, data=d, family=binomial(link='logit'), nAGQ=1,start=start,
    control=glmerControl(optimizer=optimizer, optCtrl=if(optimizer=='bobyqa')
      if(strict) list(maxfun=200000,rhoend=1e-8) else list(maxfun=200000) else
      if(strict) list(maxeval=200000,xtol_abs=1e-8,ftol_abs=1e-8) else list(maxeval=200000))),
    warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
  H <- fit@optinfo$derivs$Hessian; g <- fit@optinfo$derivs$gradient
  scaled <- tryCatch(max(abs(solve(chol(H),g))),error=function(e) NA_real_)
  diag <- list(optimizer=optimizer,strict_tolerances=strict,restarted=!is.null(start),warnings=warnings,messages=fit@optinfo$conv$lme4$messages,
    optimizer_code=fit@optinfo$conv$opt,logLik=as.numeric(logLik(fit)),singular=isSingular(fit,tol=1e-4),
    max_scaled_gradient=scaled,min_hessian_eigenvalue=if(is.null(H)) NA_real_ else min(eigen(H,symmetric=TRUE,only.values=TRUE)$values))
  list(fit=fit,diag=diag)
}
fit_checked <- function(formula,d) {
  first <- fit_once(formula,d,'bobyqa'); attempts <- list(first$diag)
  bad_messages <- first$diag$messages[!grepl('singular',first$diag$messages,ignore.case=TRUE)]
  if(length(bad_messages)>0 || any(first$diag$optimizer_code!=0) ||
     !is.finite(first$diag$max_scaled_gradient) || first$diag$max_scaled_gradient>.002 || first$diag$min_hessian_eigenvalue<=0) {
    second <- fit_once(formula,d,'nloptwrap'); attempts <- c(attempts,list(second$diag))
    # Prefer a numerically satisfactory fit; among satisfactory attempts prefer higher likelihood.
    good <- function(x) is.finite(x$diag$max_scaled_gradient) && x$diag$max_scaled_gradient<=.002 &&
      is.finite(x$diag$min_hessian_eigenvalue) && x$diag$min_hessian_eigenvalue>0 && all(x$diag$optimizer_code==0) &&
      length(x$diag$messages[!grepl('singular',x$diag$messages,ignore.case=TRUE)])==0
    if((good(second)&&!good(first)) || (good(second)==good(first)&&second$diag$logLik>first$diag$logLik)) first <- second
    if(!good(first) && !first$diag$singular) {
      third <- fit_once(formula,d,'nloptwrap',strict=TRUE); attempts <- c(attempts,list(third$diag))
      if(good(third) || third$diag$logLik>first$diag$logLik) first <- third
      if(!good(first) && !first$diag$singular) {
        fourth <- fit_once(formula,d,'bobyqa',strict=TRUE,
          start=list(theta=getME(first$fit,'theta'),fixef=fixef(first$fit)))
        attempts <- c(attempts,list(fourth$diag))
        if(good(fourth) || fourth$diag$logLik>first$diag$logLik) first <- fourth
      }
    }
  }
  list(fit=first$fit,chosen=first$diag,attempts=attempts)
}

for (model in args[-1]) {
  cat('START',model,format(Sys.time()),'\n');flush.console()
  dest <- file.path(out,model); dir.create(dest,showWarnings=FALSE)
  d <- dat[dat$model==model, ]
  stopifnot(nrow(d)==3408,length(unique(d$context))==16,length(unique(d$participant))==213)
  for (col in c('participant','context','item')) d[[col]] <- factor(d[[col]])
  if (sensitivity) {
    records <- list()
    omitted_contexts <- levels(d$context)
    requested_omit <- Sys.getenv('FOCUS_OMIT_CONTEXT')
    if(nzchar(requested_omit)) {
      stopifnot(requested_omit %in% omitted_contexts)
      omitted_contexts <- requested_omit
    }
    for (omit in omitted_contexts) {
      dd <- droplevels(d[d$context != omit, ])
      fit <- fit_checked(full_formula,dd)
      cc <- coef(summary(fit$fit))['prediction_z:alignment_c', ]
      records[[omit]] <- data.frame(model=model,omitted_context=omit,interaction=cc[1],se=cc[2],
        ci_low=cc[1]-qnorm(.975)*cc[2],ci_high=cc[1]+qnorm(.975)*cc[2],
        singular=fit$chosen$singular,optimizer_code=paste(fit$chosen$optimizer_code,collapse=';'),
        messages=paste(fit$chosen$messages,collapse=';'),scaled_gradient=fit$chosen$max_scaled_gradient)
      write_json(fit$attempts,file.path(dest,paste0('omit_',omit,'_diagnostics.json')),
        pretty=TRUE,auto_unbox=TRUE,null='null',digits=16)
      cat('OMITTED',model,omit,'interaction',cc[1],'\n');flush.console()
    }
    loo <- do.call(rbind,records)
    if(nzchar(requested_omit)) {
      previous <- read.csv(file.path(dest,'leave_one_context_out.csv'))
      loo <- rbind(previous[previous$omitted_context != requested_omit,],loo)
      loo <- loo[order(loo$omitted_context),]
    }
    write.csv(loo,file.path(dest,'leave_one_context_out.csv'),row.names=FALSE)
    if (model %in% c('ordering','set_top_k')) {
      reduced <- if (model=='ordering') neg ~ prediction_z * alignment_c + (1|participant) + (1|context) + (1|item) else
        neg ~ prediction_z * alignment_c + (1+prediction_z||participant) + (1|context) + (1|item)
      full <- fit_checked(reduced,d)
      null <- fit_checked(update(reduced, . ~ . - prediction_z:alignment_c),d)
      cc <- coef(summary(full$fit))['prediction_z:alignment_c', ]
      lrt <- 2*(as.numeric(logLik(full$fit))-as.numeric(logLik(null$fit)))
      stopifnot(lrt > -1e-5)
      write.csv(data.frame(model=model,formula=paste(deparse(reduced),collapse=' '),interaction=cc[1],se=cc[2],
        ci_low=cc[1]-qnorm(.975)*cc[2],ci_high=cc[1]+qnorm(.975)*cc[2],
        lrt_p=pchisq(max(0,lrt),1,lower.tail=FALSE),full_singular=full$chosen$singular,null_singular=null$chosen$singular),
        file.path(dest,'boundary_sensitivity.csv'),row.names=FALSE)
      write_json(list(full=full$attempts,null=null$attempts),file.path(dest,'boundary_sensitivity_diagnostics.json'),
        pretty=TRUE,auto_unbox=TRUE,null='null',digits=16)
      saveRDS(full$fit,file.path(dest,'boundary_sensitivity_full.rds'))
      saveRDS(null$fit,file.path(dest,'boundary_sensitivity_null.rds'))
    }
    next
  }
  full <- fit_checked(full_formula,d)
  cat('FULL',model,'logLik',as.numeric(logLik(full$fit)),'singular',isSingular(full$fit),'\n');flush.console()
  null <- fit_checked(null_formula,d)
  saveRDS(full$fit,file.path(dest,'full_model.rds'))
  saveRDS(null$fit,file.path(dest,'no_interaction_model.rds'))
  writeLines(capture.output(summary(full$fit)),file.path(dest,'full_model_summary.txt'))
  writeLines(capture.output(sessionInfo()),file.path(dest,'session_info.txt'))
  diagnostics <- list(full=list(chosen=full$chosen,attempts=full$attempts),null=list(chosen=null$chosen,attempts=null$attempts))
  write_json(diagnostics,file.path(dest,'diagnostics.json'),pretty=TRUE,auto_unbox=TRUE,null='null',digits=16)
  coef <- as.data.frame(coef(summary(full$fit))); coef$term <- rownames(coef);rownames(coef)<-NULL
  names(coef)[1:4] <- c('estimate','se','z','wald_p')
  coef$ci_low <- coef$estimate-qnorm(.975)*coef$se;coef$ci_high <- coef$estimate+qnorm(.975)*coef$se
  write.csv(coef,file.path(dest,'fixed_effects.csv'),row.names=FALSE)
  write.csv(as.data.frame(VarCorr(full$fit)),file.path(dest,'random_effects.csv'),row.names=FALSE)
  llfull<-as.numeric(logLik(full$fit));llnull<-as.numeric(logLik(null$fit));lrt<-2*(llfull-llnull)
  stopifnot(lrt > -1e-5)
  r<-coef[coef$term=='prediction_z:alignment_c',]
  result<-data.frame(model=model,n=nrow(d),participants=nlevels(d$participant),contexts=nlevels(d$context),pairs=nlevels(d$item),
    interaction=r$estimate,se=r$se,ci_low=r$ci_low,ci_high=r$ci_high,wald_p=r$wald_p,lrt_stat=max(0,lrt),lrt_df=1,
    lrt_p=pchisq(max(0,lrt),df=1,lower.tail=FALSE),full_logLik=llfull,null_logLik=llnull,
    full_singular=full$chosen$singular,null_singular=null$chosen$singular,
    full_scaled_gradient=full$chosen$max_scaled_gradient,null_scaled_gradient=null$chosen$max_scaled_gradient)
  write.csv(result,file.path(dest,'interaction.csv'),row.names=FALSE)
  b<-fixef(full$fit);V<-as.matrix(vcov(full$fit));a_mean<-unique(d$alignment-d$alignment_c)
  stopifnot(length(a_mean)==1)
  grid<-data.frame(alignment=seq(min(d$alignment),max(d$alignment),length.out=201))
  grid$alignment_c<-grid$alignment-a_mean
  grid$slope<-b['prediction_z']+b['prediction_z:alignment_c']*grid$alignment_c
  grid$se<-sqrt(V['prediction_z','prediction_z']+2*grid$alignment_c*V['prediction_z','prediction_z:alignment_c']+
    grid$alignment_c^2*V['prediction_z:alignment_c','prediction_z:alignment_c'])
  grid$ci_low<-grid$slope-qnorm(.975)*grid$se;grid$ci_high<-grid$slope+qnorm(.975)*grid$se
  write.csv(grid,file.path(dest,'slope_curve.csv'),row.names=FALSE)
  at<-quantile(unique(d[,c('context','alignment')])$alignment,c(.25,.5,.75),names=FALSE)
  slopes<-data.frame(alignment=at,alignment_c=at-a_mean)
  slopes$slope<-b['prediction_z']+b['prediction_z:alignment_c']*slopes$alignment_c
  slopes$se<-sqrt(V['prediction_z','prediction_z']+2*slopes$alignment_c*V['prediction_z','prediction_z:alignment_c']+
    slopes$alignment_c^2*V['prediction_z:alignment_c','prediction_z:alignment_c'])
  slopes$ci_low<-slopes$slope-qnorm(.975)*slopes$se;slopes$ci_high<-slopes$slope+qnorm(.975)*slopes$se
  write.csv(slopes,file.path(dest,'simple_slopes.csv'),row.names=FALSE)
  cat('DONE',model,'interaction',r$estimate,'LRT p',result$lrt_p,format(Sys.time()),'\n');flush.console()
}
