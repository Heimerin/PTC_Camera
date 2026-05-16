# Photon Transfer Curve Analyzer

Ten program analizuje serie biasów i flatów dla kamer astronomicznych metodą Photon Transfer Curve. Działa jako proste GUI w Tkinterze albo jako narzędzie wiersza poleceń.

## Instalacja

Wymagane minimum:

```bash
pip install numpy matplotlib
```

Dla plików FITS:

```bash
pip install astropy
```

Dla TIFF/PNG/JPG:

```bash
pip install imageio pillow
```

## Uruchomienie GUI

```bash
python ptc_analyzer.py
```

W GUI dodaj jedną kamerę na wiersz, wybierz folder biasów, folder flatów i folder wynikowy. Program obsługuje wiele kamer w jednym uruchomieniu.

## Uruchomienie z wiersza poleceń

```bash
python ptc_analyzer.py --no-gui \
  --camera "ASI_test" \
  --bias "/ścieżka/do/bias" \
  --flats "/ścieżka/do/flats" \
  --output "/ścieżka/do/wyników"
```

Opcjonalnie można podać ROI:

```bash
--roi 100,100,800,600
```

Tryb koloru:

```bash
--color-mode red
```

Dostępne są m.in. `mono`, `red`, `green`, `blue`, `luminance` dla plików RGB oraz tryby surowej mozaiki Bayera, np. `bayer_rggb_red`, `bayer_bggr_red`, `bayer_grbg_red`, `bayer_gbrg_red`. Dla kamer kolorowych najlepiej używać surowych, niedebayerowanych danych i wybrać tryb zgodny z układem Bayera, np. `bayer_rggb_red`. Jeżeli plik jest już RGB po debayeringu, użyj `red`, ale pamiętaj, że interpolacja debayeringu może zmieniać wariancję i zaburzać PTC.

Uwaga: ROI jest stosowany po wyborze kanału. Dla trybów Bayera oznacza to, że ROI dotyczy już wyekstrahowanej półrozdzielczej płaszczyzny koloru, a nie pełnej surowej mozaiki.

Zakres dopasowania wariancji do sygnału:

```bash
--fit-low 0.10 --fit-high 0.70
```

## Struktura danych

- Folder biasów: dowolna liczba klatek bias.
- Folder flatów: co najmniej dwie klatki dla każdego czasu ekspozycji.
- Czas ekspozycji jest czytany z nagłówków FITS `EXPTIME`, `EXPOSURE`, `EXP_TIME` itd.
- Jeśli nie ma nagłówka FITS, czas może być w nazwie pliku, np. `flat_10ms_01.fit`, `flat_0.25s_02.tif`, `cameraA_exposure_1.5s_a.fit`.

Jeśli nazewnictwo jest nietypowe, w GUI albo CLI można podać własny regex. Pierwsza grupa musi być liczbą ekspozycji, druga opcjonalna grupa może być jednostką `ms` albo `s`.

## Wyniki

Dla każdej kamery powstaje osobny folder z plikami:

- `report.md`: podsumowanie wyników i opis metody.
- `summary.json`: pełne wyniki w formacie maszynowym.
- `points.csv`: tabela punktów PTC dla każdej pary flatów.
- `ptc_loglog.png`: klasyczna krzywa PTC w skali log-log.
- `variance_fit.png`: wariancja względem sygnału z dopasowaniem.
- `nonlinearity.png`: \(K_{ADC}\) oraz nieliniowość.
- `signal_linearity.png`: liniowość sygnału względem czasu ekspozycji.

## Co jest liczone

Program liczy:

- master bias i średni offset,
- read noise z par biasów,
- średni sygnał flatów po odjęciu biasu,
- read+shot noise z różnicy dwóch flatów podzielonej przez \(\sqrt{2}\),
- shot noise po odjęciu read noise w kwadraturze,
- FPN jako składową całkowitego szumu po odjęciu składowej read+shot,
- \(K_{ADC}\) w e-/ADU oraz conversion gain w ADU/e-,
- FWC jako obserwowany pełny zakres oraz jako przybliżenie z punktu załamania PTC,
- dynamic range,
- nieliniowość \(K_{ADC}\) zgodnie z relacją `100 * (K - K_LOW) / K_LOW`,
- reszty liniowości sygnału względem czasu ekspozycji.
- nachylenie shot noise w skali log-log poniżej FWC; idealnie około 0.5,
- nachylenie FPN w skali log-log poniżej FWC; idealnie około 1.0.

Nachylenia pomagają interpretować typ nieliniowości. Dla klasycznego zachowania shot noise powinien mieć slope około 1/2, a FPN około 1. Przy V/V nonlinearity zwykle zmienia się efektywne \(K_{ADC}\), ale FPN nadal pozostaje blisko slope 1. Przy V/e- nonlinearity także FPN może odchodzić od slope 1, bo sygnał i szum nie skalują się już tak samo.

## Ważne ograniczenia

Pełna analiza rozróżniająca \(S_{ADC}\) i \(N_{ADC}\) dla nieliniowości typu V/e- wymaga dodatkowych założeń o rzeczywistym przyroście światła lub osobnej kalibracji light-gain. Ten program liczy klasyczną nieliniowość \(K_{ADC}\) z PTC oraz liniowość sygnału względem czasu ekspozycji, co jest praktycznym wariantem dla typowych serii flatów.
# PTC_Camera
