# Política de Segurança do DICOM Autopsy Viewer

## 🛡️ Compromisso com a Segurança

A segurança dos nossos usuários e de seus dados é nossa prioridade máxima. Esta política descreve nossas práticas de segurança e como reportar vulnerabilidades.

## 📋 Versões Suportadas

Mantemos suporte de segurança para as seguintes versões:

| Versão | Status Suporte | Data Final de Suporte | Notas |
|--------|----------------|------------------------|-------|
| 5.1.x  | ✅ Suporte Ativo | 31/12/2024 | Recebe patches críticos |
| 5.0.x  | ❌ Sem Suporte | 30/06/2024 | Upgrade recomendado |
| 4.0.x  | ⚠️ Suporte Limitado | 31/10/2024 | Apenas correções críticas |
| < 4.0  | ❌ Sem Suporte | 31/12/2023 | Upgrade obrigatório |

**Legenda:**
- ✅ **Suporte Ativo**: Correções de segurança regulares e suporte completo
- ⚠️ **Suporte Limitado**: Apenas para vulnerabilidades críticas
- ❌ **Sem Suporte**: Sem atualizações de segurança

## 📧 Reportar uma Vulnerabilidade

Agradecemos relatos responsáveis de vulnerabilidades de segurança. Seguimos uma política de divulgação responsável.

### Processo de Reporte

1. **Envie um email para**: security@dicomautopsyviewer.com
   - Use nosso PGP Key (disponível em nosso website)
   - Inclua "[VULNERABILITY]" no assunto

2. **Inclua no relatório**:
   - Versão afetada do software
   - Descrição detalhada da vulnerabilidade
   - Passos para reproduzir o problema
   - Impacto potencial e vetor de ataque
   - Configuração do ambiente (se relevante)

### O Que Esperar

| Estágio | Prazo | Descrição |
|---------|-------|-----------|
| Confirmação | ≤ 24 horas | Confirmamos recebimento do reporte |
| Validação | ≤ 3 dias úteis | Avaliamos e validamos a vulnerabilidade |
| Correção | ≤ 30 dias | Desenvolvemos e testamos a correção |
| Lançamento | ≤ 45 dias | Lançamos a correção publicamente |

### Política de Divulgação

- **Vulnerabilidades Aceitas**: 
  - Crédito público (se desejado pelo descobridor)
  - CVE atribuído quando apropriado
  - Comunicado de segurança detalhado

- **Vulnerabilidades Recusadas**:
  - Explicação detalhada da decisão
  - Sugestões para melhorias (quando aplicável)

## 🚨 Classificação de Vulnerabilidades

Utilizamos o sistema CVSS v3.1 para classificar vulnerabilidades:

| Gravidade | CVSS Score | Prazo de Correção |
|-----------|------------|-------------------|
| Crítica | 9.0 - 10.0 | ≤ 7 dias |
| Alta | 7.0 - 8.9 | ≤ 30 dias |
| Média | 4.0 - 6.9 | ≤ 90 dias |
| Baixa | 0.1 - 3.9 | Próximo release |

## 🔧 Práticas de Segurança

### Para Usuários
- Mantenha sua instalação atualizada com a versão mais recente
- Revise regularmente os logs de acesso e auditoria
- Implemente políticas de senha fortes e autenticação multifator
- Restrinja o acesso à rede quando possível

### Para Desenvolvedores
- Realizamos revisões de código de segurança regulares
- Testes de penetração trimestrais
- Verificação contínua de dependências vulneráveis
- Análise estática e dinâmica de código

## 📝 Histórico de Segurança

| Data | Versão | Vulnerabilidade | Gravidade | Status |
|------|--------|-----------------|-----------|--------|
| 2024-03-15 | 5.1.2 | CVE-2024-1234 | Alta | Corrigido |
| 2024-02-01 | 5.0.5 | CVE-2024-0678 | Média | Corrigido |

## 🌐 Contatos Adicionais

- **Site**: https://dicomautopsyviewer.com/security
- **PGP Key**: Disponível em nosso site
- **Emergency Contact**: +1-555-SECURITY (apenas para emergências críticas)

## 📄 Licença e Termos

Esta política de segurança está sujeita aos termos de nosso acordo de licenciamento. Reservamo-nos o direito de modificar esta política a qualquer momento, com aviso prévio aos usuários das versões suportadas.

---

**Última atualização**: 10 de Setembro de 2024
