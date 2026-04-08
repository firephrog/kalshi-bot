//import kalshi
import kalshiPkg from 'kalshi-typescript';
const kalshi = kalshiPkg.default || kalshiPkg;
const { Configuration, ExchangeApi, MarketApi, PortfolioApi } = kalshi;

//other imports
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

//kalshi setup
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const privateKey = fs.readFileSync(path.join(__dirname, 'cheeseburgeristasty.pem'), 'utf8');
const config = new Configuration({
    apiKey: '0d84f453-7d85-4d52-afae-63504b432041',
    privateKeyPem: privateKey,
    basePath: 'https://demo-api.kalshi.co/trade-api/v2' 
});

const exchangeApi = new ExchangeApi(config);
const marketsApi = new MarketApi(config);
const portfolioApi = new PortfolioApi(config);

//test function
async function check() {
    try {
        const response = await exchangeApi.getExchangeStatus();
        console.log('Success! Status:', response.data);
        
        // market api test
        const markets = await marketsApi.getMarkets({ limit: 5 });
        console.log('Successfully fetched markets!');
    } catch (e) {
        console.error('Bot Error:', e.message);
    }
}
check();